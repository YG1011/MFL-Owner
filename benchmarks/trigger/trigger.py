"""Trigger dataset helpers.

The upstream training script relies on two lightweight utilities for preparing
trigger data.  The implementations provided here intentionally avoid heavy
framework dependencies so that the code works in both research and production
setups.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def category_freq(labels: Iterable[int]) -> Counter:
    """Compute category frequency statistics.

    Parameters
    ----------
    labels:
        Sequence of class identifiers.  ``torch.Tensor`` and ``numpy.ndarray``
        inputs are accepted in addition to Python iterables.

    Returns
    -------
    collections.Counter
        Mapping from label to occurrence count.
    """

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
    """Load trigger images from ``image_root``.

    The loader scans ``image_root`` for image files (PNG/JPEG/WebP/BMP).  Images
    are returned as a tensor with shape ``[N, 3, H, W]`` in ``float32`` format
    scaled to ``[0, 1]`` when ``normalise`` is ``True``.
    """

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


__all__ = ["category_freq", "load_trigger_images"]
