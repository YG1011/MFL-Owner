# training.py
"""Train / update watermark trigger alignment matrices (static + dynamic)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from embedding import EmbeddingLoadError, get_emb
from options import get_parser_args

# 你方法用到的工具（按你现有 trigger.py 提供）
from trigger import (
    procrustes_rotation, apply_rotation,
    load_basis, load_Mi, whitebox_distance,
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
    batch_size_arg = args.get("batch_size")

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
    save_dir = output_root.expanduser() / method_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 白盒指标准备
    wb_report = bool(args.get("wb_report"))
    U_cache = V_cache = None

    for local_idx, client_id in enumerate(client_ids):
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

            # 读取上一轮 W，或从 I 开始
            matrix_path = save_dir / f"trigger_mat_c{client_id}_{trigger_num}_{method}.pth"
            if matrix_path.exists() and args.get("init_from_existing"):
                W_old = torch.load(matrix_path, map_location=device).to(device).float()
                print(f"==> Client {client_id}: loaded previous W from {matrix_path}")
            else:
                W_old = torch.eye(dim, device=device)
                print(f"==> Client {client_id}: start from identity W (no existing matrix or not requested).")

            # Procrustes：R* 使得 E_new R ≈ E_old
            R_star = procrustes_rotation(E_new, E_old)
            W_new = apply_rotation(W_old, R_star).to(device)

            # 可选：一次正交回拉，数值更稳
            with torch.no_grad():
                W_new -= beta * (W_new @ W_new.t() @ W_new - W_new)

            torch.save(W_new.detach().cpu(), matrix_path)
            print(f"[Dynamic] Client {client_id}: saved updated matrix to {matrix_path}")

            # 正确的动态验证：新@W_new vs 旧@W_old
            print(f"[DEBUG] client {client_id}: "
                  f"equal(E_new,E_old)? {torch.allclose(E_new, E_old)} "
                  f"max|diff|={(E_new - E_old).abs().max().item():.6f}")
            _print_dynamic_verification(W_new, W_old, E_new, E_old)

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
                )
                torch.save(matrix.cpu(), matrix_path)
                print(f"Saved matrix to {matrix_path}")

            _print_verification(matrix.to(device), images.to(device), texts.to(device))

        # -----------------------------
        # 白盒指标（可选）
        # -----------------------------
        if wb_report:
            dim_w = matrix.shape[0]
            if U_cache is None:
                U_cache = load_basis(args.get("wb_U"), dim_w, device)
            if V_cache is None:
                V_cache = load_basis(args.get("wb_V"), dim_w, device)

            Mi = load_Mi(args.get("wb_M_dir"), client_id, dim_w, device)
            if Mi is not None:
                d_wb = whitebox_distance(matrix.to(device), U_cache, V_cache, Mi)
                print(f"[WhiteBox] client {client_id}: D_wb = {d_wb:.6f}")
            else:
                print(f"[WhiteBox] client {client_id}: Mi not provided, skip.")

if __name__ == "__main__":
    main()
