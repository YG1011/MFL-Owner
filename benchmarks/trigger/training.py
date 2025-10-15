"""Train the watermark trigger alignment matrices for each client."""
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
            diff = image_batch @ matrix.t() - text_batch @ matrix.t()
            loss = torch.norm(diff, p="fro") / math.sqrt(image_batch.shape[0])
            loss.backward()

            torch.nn.utils.clip_grad_value_(matrix, clip_value=0.5)
            optimiser.step()

            if method != "random":
                with torch.no_grad():
                    matrix -= beta * (matrix @ matrix.t() @ matrix - matrix)
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / max(1, len(loader))
            print(f"Epoch {epoch:04d}: loss={avg_loss:.6f}")

    return matrix.detach().cpu()


def _print_verification(matrix: torch.Tensor, images: torch.Tensor, texts: torch.Tensor) -> None:
    with torch.no_grad():
        images = images.to(matrix.device)
        texts = texts.to(matrix.device)
        projected_images = images @ matrix.t()
        projected_texts = texts @ matrix.t()

        images_norm = F.normalize(projected_images, dim=-1)
        texts_norm = F.normalize(projected_texts, dim=-1)

        origin_cos, origin_l2 = _cosine_and_l2(images, texts)
        trigger_cos, trigger_l2 = _cosine_and_l2(images_norm, texts_norm)

        print(f"Origin:     cos={origin_cos.mean():.6f}, l2={origin_l2.mean():.6f}")
        print(f"Watermark:  cos={trigger_cos.mean():.6f}, l2={trigger_l2.mean():.6f}")
        print(f"Delta cos:  {trigger_cos.mean() - origin_cos.mean():.6f}")
        print(f"Delta l2:   {origin_l2.mean() - trigger_l2.mean():.6f}")



def main(argv: Iterable[str] | None = None) -> None:
    args = get_parser_args(argv)

    device = _select_device(str(args["device"]))
    print(f"Using device: {device}")

    data_root = args.get("data_root")
    if not data_root:
        raise SystemExit(
            "[ERROR] --data_root is required. Provide it via the command line or set TRIGGER_DATA_ROOT."
        )

    client_ids = args["client_list"]
    trigger_num = int(args["trigger_num"])
    method = str(args["method"])
    epochs = int(args["epochs"])
    lr = float(args["lr"])
    beta = float(args["beta"])

    batch_size_arg = args.get("batch_size")

    try:
        image_embeds, text_embeds = get_emb(
            len(client_ids),
            data_root,
            client_list=client_ids,
            image_key=str(args["image_key"]),
            text_key=str(args["text_key"]),
        )
    except EmbeddingLoadError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    output_root = Path(str(args["output_dir"]))
    method_dir_name = method if method.endswith("_w") else f"{method}_w"
    save_dir = output_root.expanduser() / method_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, client_id in enumerate(client_ids):
        images = image_embeds[local_idx].float()
        texts = text_embeds[local_idx].float()

        if images.shape != texts.shape:
            raise SystemExit(
                f"[ERROR] Client {client_id} has mismatched embeddings: {tuple(images.shape)} vs {tuple(texts.shape)}"
            )

        dim = images.shape[1]
        inferred_trigger_num = images.shape[0]
        if inferred_trigger_num != trigger_num:
            print(
                f"[WARN] Trigger count mismatch for client {client_id}:"
                f" expected {trigger_num}, found {inferred_trigger_num}."
            )

        matrix_path = save_dir / f"trigger_mat_c{client_id}_{trigger_num}_{method}.pth"
        if matrix_path.exists():
            print(f"==> Client {client_id}: reusing existing matrix at {matrix_path}")
            matrix = torch.load(matrix_path, map_location=device)
        else:
            print(f"==> Client {client_id}: training alignment matrix (dim={dim})")
            batch_size = batch_size_arg or _default_batch_size(inferred_trigger_num)
            matrix = _train_single_client(
                images,
                texts,
                device=device,
                epochs=epochs,
                lr=lr,
                beta=beta,
                method=method,
                batch_size=batch_size,
            )
            torch.save(matrix.cpu(), matrix_path)
            print(f"Saved matrix to {matrix_path}")

        _print_verification(matrix.to(device), images.to(device), texts.to(device))


if __name__ == "__main__":
    main()

