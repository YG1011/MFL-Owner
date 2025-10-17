# options.py
"""Command-line options for trigger matrix training."""

from __future__ import annotations
import argparse
import os
from typing import Dict, Iterable, List, Optional

def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default

def _parse_client_list(raw: Optional[Iterable[int]], *, fallback: int) -> List[int]:
    if raw:
        # 去重且保持顺序
        return list(dict.fromkeys(int(idx) for idx in raw))
    env_override = os.environ.get("TRIGGER_CLIENT_NUM")
    if env_override is not None:
        try:
            fallback = int(env_override)
        except ValueError:
            pass
    return list(range(fallback))

def get_parser_args(argv: Optional[Iterable[str]] = None) -> Dict[str, object]:
    parser = argparse.ArgumentParser(description="Train the watermark trigger matrix")

    # === 训练/数据相关 ===
    parser.add_argument("--trigger_num", type=int,
                        default=_env_int("TRIGGER_NUM", 512),
                        help="Number of trigger samples per client (default: 512).")
    parser.add_argument("--method", type=str,
                        default=os.environ.get("TRIGGER_METHOD", "noisy"),
                        choices=["noisy", "random", "orthogonal", "plain", "dynamic"],
                        help="Training strategy for the trigger alignment matrix.")
    parser.add_argument("--client_list", type=int, nargs="*",
                        default=None,
                        help="Explicit list of client identifiers to load.")
    parser.add_argument("--device", type=str,
                        default=os.environ.get("TRIGGER_DEVICE", "cuda"),
                        help="Computation device to use: 'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--data_root", type=str,
                        default=os.environ.get("TRIGGER_DATA_ROOT"),
                        help="Root directory containing trigger embeddings.")

    # 支持 image_key / text_key 两种书写（下划线或中划线）
    parser.add_argument("--image_key", "--image-key", dest="image_key",
                        default=os.environ.get("TRIGGER_IMAGE_KEY", "image"),
                        help="Key used for image embeddings in dict/npz files.")
    parser.add_argument("--text_key", "--text-key", dest="text_key",
                        default=os.environ.get("TRIGGER_TEXT_KEY", "text"),
                        help="Key used for text embeddings in dict/npz files.")

    # === 你命令里用到但原文件缺失的参数 ===
    parser.add_argument("--epochs", type=int,
                        default=_env_int("TRIGGER_EPOCHS", 1000),
                        help="Number of training epochs (default: 1000).")
    parser.add_argument("--lr", type=float,
                        default=_env_float("TRIGGER_LR", 1e-3),
                        help="Learning rate for Adam (default: 1e-3).")
    parser.add_argument("--beta", type=float,
                        default=_env_float("TRIGGER_BETA", 0.05),
                        help="Step size of orthogonal regularization (default: 0.05).")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int,
                        default=_env_int("TRIGGER_BATCH_SIZE", None),
                        help="Mini-batch size for training. Default: min(64, N).")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str,
                        default=os.environ.get("TRIGGER_OUTPUT_DIR", "/root/trigger/results/w_matrix"),
                        help="Directory to save trained matrices.")

    # === 动态水印相关 ===
    parser.add_argument("--prev_data_root", type=str, default=os.environ.get("TRIGGER_PREV_DATA_ROOT"),
                        help="Root of previous-round trigger embeddings (E^(t)) for dynamic update.")
    parser.add_argument("--new_data_root", type=str, default=os.environ.get("TRIGGER_NEW_DATA_ROOT"),
                        help="Root of current-round trigger embeddings (E^(t+1)) for dynamic update.")
    parser.add_argument("--procrustes_every", type=int,
                        default=_env_int("TRIGGER_PROCRUSTES_EVERY", 1),
                        help="Apply Procrustes update every K rounds (default: 1).")
    parser.add_argument("--init_from_existing", action="store_true",
                        help="If set, load existing Wi and update it; otherwise start from R*.")

    # === 白盒验证相关（先做指标计算） ===
    parser.add_argument("--wb_U", type=str, default=os.environ.get("WBOX_U"),
                        help="Path to orthogonal basis U (pt/pth/npy/npz). Use identity if absent.")
    parser.add_argument("--wb_V", type=str, default=os.environ.get("WBOX_V"),
                        help="Path to orthogonal basis V (pt/pth/npy/npz). Use identity if absent.")
    parser.add_argument("--wb_M_dir", type=str, default=os.environ.get("WBOX_M_DIR"),
                        help="Directory containing Mi per client: Mi_client{c}.pt")
    parser.add_argument("--wb_report", action="store_true",
                        help="Report white-box distance D_wb(i) after training/updating.")


    args = parser.parse_args(args=list(argv) if argv is not None else None)

    client_list = _parse_client_list(args.client_list, fallback=5)
    parsed: Dict[str, object] = vars(args)
    parsed["client_list"] = client_list
    parsed["client_num"] = len(client_list)
    return parsed

__all__ = ["get_parser_args"]
