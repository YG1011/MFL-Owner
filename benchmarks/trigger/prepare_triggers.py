"""Utilities to encode trigger embeddings using trained watermark matrices."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

from embedding import EmbeddingLoadError, get_emb
from trigger import blackbox_statistics, encode_targets, ensure_dir, load_matrix


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    value = os.environ.get(name)
    return value if value is not None else default


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_client_list(raw: Optional[Sequence[int]], *, fallback: int) -> List[int]:
    if raw:
        return list(dict.fromkeys(int(idx) for idx in raw))
    env_override = os.environ.get("TRIGGER_CLIENT_NUM")
    if env_override is not None:
        try:
            fallback = int(env_override)
        except ValueError:
            pass
    return list(range(fallback))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode trigger embeddings with trained watermark matrices.")

    parser.add_argument("--data_root", type=str,
                        default=_env_str("TRIGGER_DATA_ROOT", None),
                        help="Root directory containing trigger embeddings.")
    parser.add_argument("--weights_dir", type=str,
                        default=_env_str("TRIGGER_OUTPUT_DIR", "/root/trigger/results/w_matrix"),
                        help="Directory where trained trigger matrices are stored.")
    parser.add_argument("--output_dir", type=str,
                        default=_env_str("TRIGGER_TARGET_DIR", None),
                        help="Destination directory to save encoded trigger targets."
                             " Defaults to <weights_dir>/<method>/targets if omitted.")
    parser.add_argument("--method", type=str,
                        default=_env_str("TRIGGER_METHOD", "noisy"),
                        help="Training method tag used when saving matrices (e.g. 'noisy').")
    parser.add_argument("--trigger_num", type=int,
                        default=_env_int("TRIGGER_NUM", 512),
                        help="Number of trigger samples per client (used for filename resolution).")
    parser.add_argument("--client_list", type=int, nargs="*", default=None,
                        help="Explicit client identifiers to process. Defaults to range(TRIGGER_CLIENT_NUM).")
    parser.add_argument("--device", type=str,
                        default=_env_str("TRIGGER_DEVICE", "cpu"),
                        help="Computation device to use for encoding (cuda/cpu/auto).")
    parser.add_argument("--image_key", "--image-key", dest="image_key", type=str,
                        default=_env_str("TRIGGER_IMAGE_KEY", "image"),
                        help="Key used for image embeddings inside dict/npz files.")
    parser.add_argument("--text_key", "--text-key", dest="text_key", type=str,
                        default=_env_str("TRIGGER_TEXT_KEY", "text"),
                        help="Key used for text embeddings inside dict/npz files.")
    parser.add_argument("--normalise", dest="normalise", action="store_true",
                        help="Normalise encoded targets after projection (default).")
    parser.add_argument("--no-normalise", dest="normalise", action="store_false",
                        help="Disable normalisation of encoded targets.")
    parser.set_defaults(normalise=True)
    parser.add_argument("--report", action="store_true",
                        help="Print black-box similarity statistics for sanity checks.")

    return parser


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _matrix_path(weights_dir: Path, method: str, trigger_num: int, client_id: int) -> Path:
    method_dir = method if method.endswith("_w") else f"{method}_w"
    return weights_dir / method_dir / f"trigger_mat_c{client_id}_{trigger_num}_{method}.pth"


def _save_encoded(output_dir: Path, client_id: int, tensor: torch.Tensor) -> None:
    ensure_dir(output_dir)
    path = output_dir / f"B_client{client_id}.pt"
    torch.save(tensor.cpu(), path)
    print(f"Saved encoded triggers for client {client_id} to {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_clients(
    *,
    data_root: str,
    weights_dir: Path,
    output_dir: Path,
    client_ids: Iterable[int],
    trigger_num: int,
    method: str,
    device: torch.device,
    image_key: str,
    text_key: str,
    normalise: bool,
    report: bool,
) -> None:
    try:
        client_ids = list(client_ids)
        image_embeds, text_embeds = get_emb(
            len(client_ids), data_root,
            client_list=client_ids,
            image_key=image_key,
            text_key=text_key,
        )
    except EmbeddingLoadError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    for local_idx, client_id in enumerate(client_ids):
        matrix_file = _matrix_path(weights_dir, method, trigger_num, client_id)
        if not matrix_file.exists():
            raise SystemExit(f"[ERROR] Matrix file not found: {matrix_file}")

        W = load_matrix(matrix_file, map_location=device).to(device)
        embeddings = image_embeds[local_idx].to(device)

        encoded = encode_targets(embeddings, W, normalise=normalise)
        _save_encoded(output_dir, client_id, encoded)

        if report:
            texts = text_embeds[local_idx].to(device)
            stats = blackbox_statistics(embeddings, texts, matrix=W)
            print(
                f"[Report] client {client_id}: cos={stats['cos_mean']:.6f}±{stats['cos_std']:.6f}, "
                f"l2={stats['l2_mean']:.6f}±{stats['l2_std']:.6f}"
            )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)

    data_root = args.data_root
    if not data_root:
        raise SystemExit("[ERROR] --data_root is required to locate embeddings.")

    weights_dir = Path(args.weights_dir).expanduser()
    if not weights_dir.exists():
        raise SystemExit(f"[ERROR] weights directory does not exist: {weights_dir}")

    method = str(args.method)
    trigger_num = int(args.trigger_num)

    if args.client_list is not None:
        client_ids = _parse_client_list(args.client_list, fallback=len(args.client_list))
    else:
        client_ids = _parse_client_list(None, fallback=5)

    device = _resolve_device(str(args.device))
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir is None:
        method_dir = method if method.endswith("_w") else f"{method}_w"
        output_dir = weights_dir / method_dir / "targets"
    ensure_dir(output_dir)

    encode_clients(
        data_root=data_root,
        weights_dir=weights_dir,
        output_dir=output_dir,
        client_ids=client_ids,
        trigger_num=trigger_num,
        method=method,
        device=device,
        image_key=str(args.image_key),
        text_key=str(args.text_key),
        normalise=bool(args.normalise),
        report=bool(args.report),
    )


if __name__ == "__main__":
    main()
