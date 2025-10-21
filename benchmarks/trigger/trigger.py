# trigger.py
"""Trigger dataset & watermark helpers."""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HADAMARD_CACHE: Dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

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


def hadamard_matrix(order: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    """Return a normalised Hadamard matrix of ``order`` when available."""

    if order <= 0 or order & (order - 1):  # not a power of two
        return None

    cache_key = (order, device, dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    size = 1
    while size < order:
        top = torch.cat([H, H], dim=1)
        bottom = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bottom], dim=0)
        size *= 2

    H = H / math.sqrt(order)
    _HADAMARD_CACHE[cache_key] = H
    return H


def build_whitebox_transform(
    Mi: torch.Tensor,
    *,
    mode: str = "diag_hadamard",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Construct linear transforms applied inside the white-box penalty.

    The routine returns left/right matrices ``(L, R)`` so that a projection ``P``
    is transformed as ``L @ P @ R`` before measuring distances.  ``mode``
    controls which transforms are included:

    ``identity``
        Leave the projection untouched.
    ``diag``
        Scale by the absolute value of ``Mi``'s diagonal entries.
    ``hadamard``
        Apply a normalised Hadamard transform (only when the rank is a power of
        two).
    ``diag_hadamard``
        Apply diagonal scaling followed by the Hadamard transform.
    """

    mode = mode.lower()
    if mode not in {"identity", "diag", "hadamard", "diag_hadamard"}:
        raise ValueError(f"Unsupported white-box transform mode: {mode}")

    rank = Mi.shape[0]
    device = Mi.device
    dtype = Mi.dtype

    left = torch.eye(rank, device=device, dtype=dtype)
    right = torch.eye(rank, device=device, dtype=dtype)
    changed = False

    if mode in {"diag", "diag_hadamard"}:
        diag = torch.diagonal(Mi).abs()
        if torch.count_nonzero(diag) > 0:
            diag = diag / diag.norm(p=2).clamp_min(1e-6)
            D = torch.diag(diag)
            left = D @ left
            right = right @ D
            changed = True

    if mode in {"hadamard", "diag_hadamard"}:
        H = hadamard_matrix(rank, device=device, dtype=dtype)
        if H is not None:
            left = H @ left
            right = right @ H
            changed = True

    if not changed:
        return None, None

    return left, right


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
    *,
    margin: float = 0.0,
    contrast_weight: float = 0.0,
    contrast_targets: Sequence[torch.Tensor] | None = None,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single proximal step towards the fingerprint target ``Mi``."""

    if gamma <= 0 or Mi is None:
        return W

    contrast_targets = tuple(contrast_targets or ())

    proj = U.T @ W @ V
    if left is not None:
        proj = left @ proj
    if right is not None:
        proj = proj @ right

    diff = proj - Mi

    back = diff
    if left is not None:
        back = left.T @ back
    if right is not None:
        back = back @ right.T
    grad = U @ back @ V.T
    W = (W - 2.0 * gamma * grad).to(W.dtype)

    if margin > 0 and contrast_weight > 0 and contrast_targets:
        for other in contrast_targets:
            diff_other = proj - other
            dist = torch.linalg.norm(diff_other, ord="fro")
            dist_value = float(dist)
            if dist_value < margin:
                denom = max(dist_value, 1e-6)
                direction = diff_other / denom
                back_dir = direction
                if left is not None:
                    back_dir = left.T @ back_dir
                if right is not None:
                    back_dir = back_dir @ right.T
                grad_contrast = U @ back_dir @ V.T
                W = (W + gamma * contrast_weight * grad_contrast).to(W.dtype)

    return W


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

def load_basis(
    path: Optional[str],
    dim: int,
    device: torch.device,
    *,
    rank: Optional[int] = None,
) -> torch.Tensor:
    """Load an orthonormal basis ``U``/``V`` for the white-box subspace.

    Parameters
    ----------
    path:
        Location of the serialized basis.  When ``None`` an identity basis is
        synthesised.
    dim:
        Ambient dimensionality ``d`` of the trigger matrix.
    device:
        Target device for the returned tensor.
    rank:
        Optional rank ``k`` of the white-box subspace.  When provided the
        routine returns only the leading ``k`` columns of the orthonormal basis,
        matching the low-rank formulation in the paper.  ``None`` keeps all
        columns.
    """

    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive when specified")

    if not path:
        basis = torch.eye(dim, device=device)
    else:
        B = _load_tensor(path, map_location=device).to(device).float()
        if B.ndim != 2 or B.shape[0] != dim:
            raise ValueError(
                f"Basis at {path} must be 2-D with first dim={dim}, got {tuple(B.shape)}"
            )
        # ``torch.linalg.qr`` with ``mode='reduced'`` yields ``(dim, min(dim, n))``.
        Q, _ = torch.linalg.qr(B, mode="reduced")
        basis = Q

    if rank is not None:
        if rank > basis.shape[1]:
            raise ValueError(
                f"Requested rank {rank} exceeds available columns {basis.shape[1]} in basis"
            )
        basis = basis[:, :rank]

    return basis


def load_Mi(
    m_dir: Optional[str],
    client_id: int,
    dim: int,
    device: torch.device,
    *,
    rank: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Load client-specific fingerprint ``M_i`` if present.

    ``M_i`` is expected to be a ``rank × rank`` tensor that lives in the
    low-rank white-box subspace.  For backward compatibility the loader also
    accepts ``dim × dim`` matrices and projects them onto the leading ``rank``
    components when a rank is specified.
    """

    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive when specified")

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
                if M.ndim != 2:
                    raise ValueError(f"Mi at {p} must be 2-D, got {tuple(M.shape)}")
                if rank is None:
                    if M.shape != (dim, dim):
                        raise ValueError(
                            f"Mi at {p} must be [{dim},{dim}] when rank is not specified, got {tuple(M.shape)}"
                        )
                    return M
                # Rank-aware path: accept either ``(rank, rank)`` directly or
                # ``(dim, dim)`` which we project onto the leading ``rank``
                # components for compatibility with older assets.
                if M.shape == (rank, rank):
                    return M
                if M.shape == (dim, dim):
                    return M[:rank, :rank]
                raise ValueError(
                    f"Mi at {p} must be ({rank},{rank}) or ({dim},{dim}), got {tuple(M.shape)}"
                )
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
    "hadamard_matrix",
    "build_whitebox_transform",
    # 白盒
    "load_basis",
    "load_Mi",
    "whitebox_distance",
]
