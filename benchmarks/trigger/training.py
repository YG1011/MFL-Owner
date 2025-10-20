# training.py
"""Train / update watermark trigger alignment matrices (static + dynamic)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple, Optional, List
import glob

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from embedding import EmbeddingLoadError, get_emb
from options import get_parser_args

# 你方法用到的工具（按你现有 trigger.py 提供）
from trigger import (
    procrustes_rotation,
    dynamic_watermark_update,
    time_consistency_update,
    apply_whitebox_penalty,
    encode_targets,
    blackbox_statistics,
    load_basis,
    load_Mi,
    whitebox_distance,
)

# -----------------------------
# 小工具
# -----------------------------

def _select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)

def _default_batch_size(count: int) -> int:
    return min(64, count) if count > 0 else 1

def _cosine_and_l2(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cosine = F.cosine_similarity(a, b, dim=1, eps=1e-6)
    l2 = F.pairwise_distance(a, b, p=2)
    return cosine, l2

def _find_prev_matrix(
    prev_dir: Path,
    client_id: int,
    trigger_num: int,
    preferred_method: Optional[str] = None
) -> Optional[Path]:
    """
    在 prev_dir 下尝试多种命名，找到上一轮的 W 路径。
    返回第一个存在的路径；找不到则返回 None。
    """
    candidates: List[Path] = []

    # 明确后缀优先
    if preferred_method:
        candidates.append(prev_dir / f"trigger_mat_c{client_id}_{trigger_num}_{preferred_method}.pth")

    # 常见后缀兜底
    for suf in ["dynamic", "plain", "orthogonal", "random", "noisy"]:
        p = prev_dir / f"trigger_mat_c{client_id}_{trigger_num}_{suf}.pth"
        if p not in candidates:
            candidates.append(p)

    # 最后通配匹配（例如自定义命名）
    wildcard_list = sorted(prev_dir.glob(f"trigger_mat_c{client_id}_{trigger_num}_*.pth"))
    for p in wildcard_list:
        if p not in candidates:
            candidates.append(p)

    for p in candidates:
        if p.exists():
            return p
    return None

# -----------------------------
# 静态训练一个 Wi
# -----------------------------

def _train_single_client(
    images: torch.Tensor,
    texts: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    beta: float,
    method: str,
    batch_size: int,
    whitebox: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None,
) -> torch.Tensor:
    dataset = TensorDataset(images.cpu(), texts.cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dim = images.shape[1]
    matrix = nn.Parameter(torch.empty(dim, dim, device=device))
    nn.init.orthogonal_(matrix)

    optimiser = optim.Adam([matrix], lr=lr, eps=1e-5)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for image_batch, text_batch in loader:
            image_batch = image_batch.to(device)
            text_batch = text_batch.to(device)

            optimiser.zero_grad(set_to_none=True)
            # || W X - W T ||_F
            diff = image_batch @ matrix.t() - text_batch @ matrix.t()
            loss = torch.norm(diff, p="fro") / math.sqrt(image_batch.shape[0])

            if whitebox is not None:
                U, V, Mi, gamma = whitebox
                if Mi is not None and gamma > 0:
                    proj = U.t() @ matrix @ V - Mi
                    loss = loss + gamma * torch.norm(proj, p="fro") ** 2

            loss.backward()

            torch.nn.utils.clip_grad_value_(matrix, clip_value=0.5)
            optimiser.step()

            if method != "random":
                # 正交回拉： W ← W − β (W W^T W − W)
                with torch.no_grad():
                    matrix -= beta * (matrix @ matrix.t() @ matrix - matrix)

            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / max(1, len(loader))
            print(f"Epoch {epoch:04d}: loss={avg_loss:.6f}")

    return matrix.detach().cpu()

# -----------------------------
# 黑盒验证日志（静态）
# -----------------------------

@torch.no_grad()
def _print_verification(matrix: torch.Tensor, images: torch.Tensor, texts: torch.Tensor) -> None:
    images = images.to(matrix.device)
    texts = texts.to(matrix.device)

    # 直接用未投影/已投影两组做对比
    projected_images = images @ matrix.t()
    projected_texts  = texts  @ matrix.t()

    images_norm = F.normalize(projected_images, dim=-1)
    texts_norm  = F.normalize(projected_texts,  dim=-1)

    origin_cos, origin_l2   = _cosine_and_l2(images, texts)            # 原始差异
    trigger_cos, trigger_l2 = _cosine_and_l2(images_norm, texts_norm)  # 应用 W 后差异

    print(f"Origin:     cos={origin_cos.mean():.6f}, l2={origin_l2.mean():.6f}")
    print(f"Watermark:  cos={trigger_cos.mean():.6f}, l2={trigger_l2.mean():.6f}")
    print(f"Delta cos:  {trigger_cos.mean() - origin_cos.mean():.6f}")
    print(f"Delta l2:   {origin_l2.mean() - trigger_l2.mean():.6f}")

# -----------------------------
# 黑盒验证日志（动态：新@W_new vs 旧@W_old）
# -----------------------------

@torch.no_grad()
def _print_dynamic_verification(
    W_new: torch.Tensor,
    W_old: torch.Tensor,
    E_new: torch.Tensor,
    E_old: torch.Tensor,
) -> None:
    # 原始（动态前）：直接比较新旧两轮的嵌入
    a0 = F.normalize(E_new, dim=-1)
    b0 = F.normalize(E_old, dim=-1)
    o_cos, o_l2 = _cosine_and_l2(a0, b0)

    # 动态后：新用 W_new，旧用 W_old
    a1 = F.normalize(E_new @ W_new.t(), dim=-1)
    b1 = F.normalize(E_old @ W_old.t(), dim=-1)
    w_cos, w_l2 = _cosine_and_l2(a1, b1)

    print(f"Origin(new vs old):                   cos={o_cos.mean():.6f}, l2={o_l2.mean():.6f}")
    print(f"Watermark(new@W_new vs old@W_old):    cos={w_cos.mean():.6f}, l2={w_l2.mean():.6f}")
    print(f"Delta cos:                            {(w_cos.mean()-o_cos.mean()):.6f}")
    print(f"Delta l2:                             {(o_l2.mean()-w_l2.mean()):.6f}")

# -----------------------------
# 入口
# -----------------------------

def main(argv: Iterable[str] | None = None) -> None:
    args = get_parser_args(argv)

    device = _select_device(str(args["device"]))
    print(f"Using device: {device}")

    # 新增：读/写目录与旧后缀参数（需要在 options.py 中添加）
    prev_w_dir_opt = args.get("prev_w_dir")
    save_w_dir_opt = args.get("save_w_dir")
    prev_w_method  = args.get("prev_w_method")

    # 输入路径
    data_root = args.get("data_root")
    prev_root = args.get("prev_data_root")
    new_root  = args.get("new_data_root")

    client_ids     = args["client_list"]
    trigger_num    = int(args["trigger_num"])
    method         = str(args["method"])
    epochs         = int(args["epochs"])
    lr             = float(args["lr"])
    beta           = float(args["beta"])
    time_lambda    = float(args.get("time_lambda", 0.0))
    time_mu        = float(args.get("time_mu", 0.0))
    batch_size_arg = args.get("batch_size")
    whitebox_gamma = float(args.get("whitebox_gamma", 0.0))
    save_targets   = bool(args.get("save_targets"))
    target_dir_opt = args.get("target_dir")

    # 动态 / 静态 模式判定
    dynamic_mode = bool(prev_root and new_root)

    if dynamic_mode:
        print("[Dynamic] Procrustes update mode enabled.")
        try:
            img_prev, txt_prev = get_emb(
                len(client_ids), prev_root,
                client_list=client_ids,
                image_key=str(args["image_key"]),
                text_key=str(args["text_key"]),
            )
            img_new, txt_new = get_emb(
                len(client_ids), new_root,
                client_list=client_ids,
                image_key=str(args["image_key"]),
                text_key=str(args["text_key"]),
            )
        except EmbeddingLoadError as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
    else:
        if not data_root:
            raise SystemExit("[ERROR] --data_root is required in static mode.")
        try:
            image_embeds, text_embeds = get_emb(
                len(client_ids), data_root,
                client_list=client_ids,
                image_key=str(args["image_key"]),
                text_key=str(args["text_key"]),
            )
        except EmbeddingLoadError as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc

    # 输出目录
    output_root = Path(str(args["output_dir"]))
    method_dir_name = method if method.endswith("_w") else f"{method}_w"

    # 允许 --save_w_dir 覆盖保存目录
    if save_w_dir_opt:
        save_dir = Path(save_w_dir_opt).expanduser()
    else:
        save_dir = output_root.expanduser() / method_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[IO] Save dir = {save_dir}")

    # 上一轮目录（若不给则退回 save_dir）
    prev_w_dir = Path(prev_w_dir_opt).expanduser() if prev_w_dir_opt else save_dir
    if prev_w_dir_opt:
        prev_w_dir.mkdir(parents=True, exist_ok=True)
    print(f"[IO] Prev-W dir = {prev_w_dir}")

    target_dir = None
    if save_targets:
        if target_dir_opt:
            target_dir = Path(str(target_dir_opt)).expanduser()
        else:
            target_dir = save_dir / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)

    # 白盒指标准备
    wb_report = bool(args.get("wb_report"))
    U_cache = V_cache = None

    for local_idx, client_id in enumerate(client_ids):
        Mi: torch.Tensor | None = None
        whitebox_payload: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None

        if dynamic_mode:
            # 用“图像端”的两轮嵌入
            E_old = img_prev[local_idx].float().to(device)  # t
            E_new = img_new[local_idx].float().to(device)   # t+1

            if E_old.shape != E_new.shape:
                raise SystemExit(
                    f"[ERROR] Client {client_id}: prev/new shapes mismatch: "
                    f"{tuple(E_old.shape)} vs {tuple(E_new.shape)}"
                )

            dim = E_old.shape[1]
            inferred_trigger_num = E_old.shape[0]
            if inferred_trigger_num != trigger_num:
                print(f"[WARN] Client {client_id}: trigger count mismatch: "
                      f"expected {trigger_num}, found {inferred_trigger_num}.")

            if whitebox_gamma > 0 or wb_report:
                if U_cache is None or U_cache.shape[0] != dim:
                    U_cache = load_basis(args.get("wb_U"), dim, device)
                if V_cache is None or V_cache.shape[0] != dim:
                    V_cache = load_basis(args.get("wb_V"), dim, device)
                Mi = load_Mi(args.get("wb_M_dir"), client_id, dim, device)
                if whitebox_gamma > 0 and Mi is None:
                    print(f"[WARN] client {client_id}: Mi not found, skip white-box penalty.")
                elif whitebox_gamma > 0 and Mi is not None:
                    whitebox_payload = (U_cache, V_cache, Mi, whitebox_gamma)

            # 读取上一轮 W（优先 prev_w_dir），或从 I 开始
            matrix_path_new = save_dir / f"trigger_mat_c{client_id}_{trigger_num}_{method}.pth"
            matrix_path_old = _find_prev_matrix(prev_w_dir, client_id, trigger_num, prev_w_method)

            if matrix_path_old is not None and args.get("init_from_existing"):
                W_old = torch.load(matrix_path_old, map_location=device).to(device).float()
                print(f"==> Client {client_id}: loaded previous W from {matrix_path_old}")
            else:
                W_old = torch.eye(dim, device=device)
                msg = "(no existing matrix found)" if matrix_path_old is None else "(init_from_existing not set)"
                print(f"==> Client {client_id}: start from identity W {msg}.")

            # Procrustes：R* 使得 E_old ≈ E_new R
            R_star = procrustes_rotation(E_old, E_new)
            W_new = dynamic_watermark_update(W_old, R_star, beta)
            W_new = time_consistency_update(
                W_new,
                W_prev=W_old,
                R=R_star,
                lambda_=time_lambda,
                mu=time_mu,
            )

            if whitebox_payload is not None:
                U_dyn, V_dyn, Mi_dyn, gamma_dyn = whitebox_payload
                W_new = apply_whitebox_penalty(W_new, U_dyn, V_dyn, Mi_dyn, gamma_dyn)

            torch.save(W_new.detach().cpu(), matrix_path_new)
            print(f"[Dynamic] Client {client_id}: saved updated matrix to {matrix_path_new}")

            # 正确的动态验证：新@W_new vs 旧@W_old
            print(f"[DEBUG] client {client_id}: "
                  f"equal(E_new,E_old)? {torch.allclose(E_new, E_old)} "
                  f"max|diff|={(E_new - E_old).abs().max().item():.6f}")
            _print_dynamic_verification(W_new, W_old, E_new, E_old)

            before_stats = blackbox_statistics(E_new, E_old)
            after_stats = blackbox_statistics(E_new @ W_new.t(), E_old @ W_old.t())
            print(
                f"[Dynamic-BlackBox] client {client_id}: "
                f"before cos={before_stats['cos_mean']:.6f}±{before_stats['cos_std']:.6f}, "
                f"after cos={after_stats['cos_mean']:.6f}±{after_stats['cos_std']:.6f}"
            )

            if save_targets and target_dir is not None:
                encoded = encode_targets(E_new, W_new)
                torch.save(encoded.cpu(), target_dir / f"B_client{client_id}.pt")

            matrix = W_new  # 供白盒指标用

        else:
            # ---------- 静态训练 ----------
            images = image_embeds[local_idx].float()
            texts  = text_embeds[local_idx].float()

            if images.shape != texts.shape:
                raise SystemExit(
                    f"[ERROR] Client {client_id} has mismatched embeddings: "
                    f"{tuple(images.shape)} vs {tuple(texts.shape)}"
                )

            dim = images.shape[1]
            inferred_trigger_num = images.shape[0]
            if inferred_trigger_num != trigger_num:
                print(f"[WARN] Trigger count mismatch for client {client_id}: "
                      f"expected {trigger_num}, found {inferred_trigger_num}.")

            if whitebox_gamma > 0 or wb_report:
                if U_cache is None or U_cache.shape[0] != dim:
                    U_cache = load_basis(args.get("wb_U"), dim, device)
                if V_cache is None or V_cache.shape[0] != dim:
                    V_cache = load_basis(args.get("wb_V"), dim, device)
                Mi = load_Mi(args.get("wb_M_dir"), client_id, dim, device)
                if whitebox_gamma > 0 and Mi is None:
                    print(f"[WARN] client {client_id}: Mi not found, skip white-box penalty.")
                elif whitebox_gamma > 0 and Mi is not None:
                    whitebox_payload = (U_cache, V_cache, Mi, whitebox_gamma)

            matrix_path = save_dir / f"trigger_mat_c{client_id}_{trigger_num}_{method}.pth"
            if matrix_path.exists():
                print(f"==> Client {client_id}: reusing existing matrix at {matrix_path}")
                matrix = torch.load(matrix_path, map_location=device)
            else:
                print(f"==> Client {client_id}: training alignment matrix (dim={dim})")
                batch_size = batch_size_arg or _default_batch_size(inferred_trigger_num)

                print(
                    f"[DEBUG] client {client_id}: "
                    f"equal(images,texts)? {torch.allclose(images, texts)} "
                    f"max|diff|={(images - texts).abs().max().item():.6f}"
                )

                matrix = _train_single_client(
                    images, texts,
                    device=device, epochs=epochs, lr=lr, beta=beta,
                    method=method, batch_size=batch_size,
                    whitebox=whitebox_payload,
                )
                torch.save(matrix.cpu(), matrix_path)
                print(f"Saved matrix to {matrix_path}")

            matrix_device = matrix.to(device)
            _print_verification(matrix_device, images.to(device), texts.to(device))

            stats = blackbox_statistics(images.to(device), texts.to(device), matrix=matrix_device)
            print(
                f"[BlackBox] client {client_id}: "
                f"cos={stats['cos_mean']:.6f}±{stats['cos_std']:.6f}, "
                f"l2={stats['l2_mean']:.6f}±{stats['l2_std']:.6f}"
            )

            if save_targets and target_dir is not None:
                encoded = encode_targets(images.to(device), matrix_device)
                torch.save(encoded.cpu(), target_dir / f"B_client{client_id}.pt")

            matrix = matrix_device

        # -----------------------------
        # 白盒指标（可选）
        # -----------------------------
        if wb_report:
            dim_w = matrix.shape[0]
            if U_cache is None or U_cache.shape[0] != dim_w:
                U_cache = load_basis(args.get("wb_U"), dim_w, device)
            if V_cache is None or V_cache.shape[0] != dim_w:
                V_cache = load_basis(args.get("wb_V"), dim_w, device)

            Mi_report = Mi if Mi is not None else load_Mi(args.get("wb_M_dir"), client_id, dim_w, device)
            if Mi_report is not None:
                d_wb = whitebox_distance(matrix.to(device), U_cache, V_cache, Mi_report)
                print(f"[WhiteBox] client {client_id}: D_wb = {d_wb:.6f}")
            else:
                print(f"[WhiteBox] client {client_id}: Mi not provided, skip.")

if __name__ == "__main__":
    main()
