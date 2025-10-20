# trigger.py
"""Trigger dataset & watermark helpers."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# -----------------------------
# 1) 与数据相关（保留原实现）
# -----------------------------

def category_freq(labels: Iterable[int]) -> Counter:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()
    return Counter(int(label) for label in labels)

def load_trigger_images(
    image_root: str,
    *,
    limit: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    normalise: bool = True,
) -> torch.Tensor:
    root = Path(os.path.expanduser(image_root))
    if not root.exists():
        raise FileNotFoundError(f"Trigger image directory '{root}' does not exist")

    supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in supported_suffixes]
    if not image_paths:
        raise FileNotFoundError(f"No image files found under '{root}'")
    if limit is not None:
        image_paths = image_paths[:limit]

    tensors: List[torch.Tensor] = []
    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if resize is not None:
                img = img.resize(resize, Image.BILINEAR)
            array = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if normalise:
            tensor = tensor / 255.0
        tensors.append(tensor)

    return torch.stack(tensors)

# -----------------------------
# 2) 通用 I/O 工具
# -----------------------------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_tensor(path: str | Path, *, key: Optional[str] = None, map_location: str | torch.device = "cpu") -> torch.Tensor:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location=map_location)
        if isinstance(obj, torch.Tensor):
            return obj.float()
        if isinstance(obj, dict):
            if key is None:
                # 默认取第一个 tensor
                for v in obj.values():
                    if isinstance(v, torch.Tensor):
                        return v.float()
                raise RuntimeError(f"No tensor found in dict at {path}")
            return torch.as_tensor(obj[key]).float()
        raise RuntimeError(f"Unsupported object in {path}: {type(obj)}")

    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return torch.as_tensor(arr, dtype=torch.float32)

    if path.suffix == ".npz":
        arr = np.load(path)
        if key is None:
            if len(arr.files) != 1:
                raise RuntimeError(f"{path} contains multiple arrays, specify key")
            key = arr.files[0]
        return torch.as_tensor(arr[key], dtype=torch.float32)

    raise RuntimeError(f"Unsupported file extension for {path}")

def save_matrix(path: str | Path, W: torch.Tensor) -> None:
    path = Path(path).expanduser()
    ensure_dir(path.parent)
    torch.save(W.detach().cpu(), path)

def load_matrix(path: str | Path, map_location: str | torch.device = "cpu") -> torch.Tensor:
    return _load_tensor(path, map_location=map_location)

# -----------------------------
# 3) 动态水印：Procrustes 旋转
# -----------------------------

@torch.no_grad()
def procrustes_rotation(E_new: torch.Tensor, E_old: torch.Tensor) -> torch.Tensor:
    """Compute the Procrustes rotation aligning ``E_new`` to ``E_old``."""

    if E_new.ndim != 2 or E_old.ndim != 2 or E_new.shape != E_old.shape:
        raise ValueError(
            "E_new and E_old must be two 2-D tensors with identical shapes for Procrustes alignment"
        )

    C = E_old.T @ E_new  # [d, d]
    U, _, Vh = torch.linalg.svd(C, full_matrices=False)
    R = U @ Vh

    # Numerical safeguard: eliminate potential reflections so that det(R) ≈ 1
    det = torch.det(R)
    if det < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vh

    return R


@torch.no_grad()
def dynamic_watermark_update(W_old: torch.Tensor, R: torch.Tensor, beta: float) -> torch.Tensor:
    """Adaptive update ``W_old`` using rotation ``R`` and orthogonal relaxation ``beta``."""

    W_rot = (W_old @ R).to(dtype=W_old.dtype)
    if beta <= 0:
        return W_rot

    gram = W_rot @ W_rot.T
    return (1.0 + beta) * W_rot - beta * gram @ W_rot


@torch.no_grad()
def time_consistency_update(
    W_candidate: torch.Tensor,
    *,
    W_prev: torch.Tensor,
    R: torch.Tensor,
    lambda_: float = 0.0,
    mu: float = 0.0,
) -> torch.Tensor:
    """Apply optional time-consistency smoothing (Eq. 7 in the spec)."""

    if lambda_ > 0:
        target = (R @ W_prev).to(W_candidate.dtype)
        W_candidate = (1.0 - lambda_) * W_candidate + lambda_ * target

    if mu > 0:
        W_candidate = (1.0 - mu) * W_candidate + mu * W_prev.to(W_candidate.dtype)

    return W_candidate


@torch.no_grad()
def apply_whitebox_penalty(
    W: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    Mi: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Single proximal step towards the fingerprint target ``Mi``."""

    if gamma <= 0:
        return W

    diff = U.T @ W @ V - Mi
    grad = U @ diff @ V.T
    return (W - 2.0 * gamma * grad).to(W.dtype)


@torch.no_grad()
def encode_targets(embeddings: torch.Tensor, W: torch.Tensor, *, normalise: bool = True) -> torch.Tensor:
    """Project trigger embeddings through ``W`` to generate dynamic targets."""

    encoded = embeddings @ W.T
    if normalise:
        encoded = F.normalize(encoded, dim=-1, eps=1e-6)
    return encoded


@torch.no_grad()
def blackbox_statistics(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    *,
    matrix: torch.Tensor | None = None,
) -> dict:
    """Compute cosine/L2 statistics before and after watermarking."""

    if matrix is not None:
        embeddings_a = embeddings_a @ matrix.T
        embeddings_b = embeddings_b @ matrix.T

    norm_a = F.normalize(embeddings_a, dim=-1, eps=1e-6)
    norm_b = F.normalize(embeddings_b, dim=-1, eps=1e-6)

    cos = F.cosine_similarity(norm_a, norm_b, dim=-1, eps=1e-6)
    l2 = F.pairwise_distance(norm_a, norm_b, p=2)

    return {
        "cos_mean": cos.mean().item(),
        "cos_std": cos.std(unbiased=False).item(),
        "l2_mean": l2.mean().item(),
        "l2_std": l2.std(unbiased=False).item(),
    }

# -----------------------------
# 4) 白盒指标/基/水印加载
# -----------------------------

def load_basis(path: Optional[str], dim: int, device: torch.device) -> torch.Tensor:
    """
    读取正交基 U/V；若 path 为空则返回 I。
    若不是严格正交，会做一次 QR 以正交化。
    """
    if not path:
        return torch.eye(dim, device=device)
    B = _load_tensor(path, map_location=device).to(device).float()
    if B.ndim != 2 or B.shape[0] != dim:
        raise ValueError(f"Basis at {path} must be 2-D with first dim={dim}, got {tuple(B.shape)}")
    # 简单正交化
    Q, _ = torch.linalg.qr(B, mode="reduced")
    return Q

def load_Mi(m_dir: Optional[str], client_id: int, dim: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    在目录里查找 Mi 文件，常见命名：
      Mi_client{c}.pt / Mi_c{c}.pt / Mi_{c}.pt / client_{c}.pt
    若不存在返回 None。
    """
    if not m_dir:
        return None
    m_dir = str(m_dir)
    candidates = [
        f"Mi_client{client_id}",
        f"Mi_c{client_id}",
        f"Mi_{client_id}",
        f"client_{client_id}",
        f"m_{client_id}",
    ]
    for base in candidates:
        for suf in (".pt", ".pth", ".npy", ".npz"):
            p = Path(m_dir).expanduser() / f"{base}{suf}"
            if p.exists():
                M = _load_tensor(p, map_location=device).to(device).float()
                if M.shape != (dim, dim):
                    raise ValueError(f"Mi at {p} must be [{dim},{dim}], got {tuple(M.shape)}")
                return M
    return None

@torch.no_grad()
def whitebox_distance(W_enc: torch.Tensor, U: torch.Tensor, V: torch.Tensor, Mi: torch.Tensor) -> float:
    """
    D_wb(i) = || U^T W_enc V - M_i ||_F
    返回标量 float
    """
    proj = U.T @ W_enc @ V
    return torch.linalg.norm(proj - Mi, ord="fro").item()

__all__ = [
    # 原有
    "category_freq",
    "load_trigger_images",
    # I/O
    "ensure_dir",
    "save_matrix",
    "load_matrix",
    # 动态水印
    "procrustes_rotation",
    "dynamic_watermark_update",
    "time_consistency_update",
    "apply_whitebox_penalty",
    "encode_targets",
    "blackbox_statistics",
    # 白盒
    "load_basis",
    "load_Mi",
    "whitebox_distance",
]
