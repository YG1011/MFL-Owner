"""Utilities for loading trigger embeddings.

The original project shipped a small helper module that loaded per-client image
and text embeddings from disk.  This re-implementation is intentionally
flexible: it supports a variety of directory layouts and file formats so that it
can work with both the authors' internal data dumps and custom datasets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# Common filename patterns for image/text embeddings.  The loader searches for
# these names (with any of the known suffixes) inside each client directory.
_IMAGE_PATTERNS = (
    "image_embeddings",
    "image_embed",
    "image",  # fallback for generic dumps
    "img",
)
_TEXT_PATTERNS = (
    "text_embeddings",
    "text_embed",
    "text",  # fallback for generic dumps
    "txt",
)
_SUFFIXES = (".pt", ".pth", ".npy", ".npz")


class EmbeddingLoadError(RuntimeError):
    """Raised when a client directory cannot be resolved into embeddings."""


def _load_combined(path: Path, *, image_key: str, text_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a file containing both image and text embeddings."""

    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            try:
                image = payload[image_key]
                text = payload[text_key]
            except KeyError as exc:
                raise EmbeddingLoadError(
                    f"Missing key '{exc.args[0]}' in {path}. Available keys: {list(payload.keys())}"
                ) from exc
        elif isinstance(payload, (list, tuple)) and len(payload) == 2:
            image, text = payload
        else:
            raise EmbeddingLoadError(
                f"Unsupported tensor format in {path}: expected dict or 2-tuple, got {type(payload)!r}"
            )
        return torch.as_tensor(image, dtype=torch.float32), torch.as_tensor(text, dtype=torch.float32)

    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=True)
        if isinstance(array, np.ndarray) and array.shape and array.dtype == object and len(array) == 2:
            image, text = array
            return torch.as_tensor(image, dtype=torch.float32), torch.as_tensor(text, dtype=torch.float32)
        raise EmbeddingLoadError(
            f"NumPy file {path} must contain a length-2 object array with image/text embeddings."
        )

    if path.suffix == ".npz":
        array = np.load(path)
        if image_key not in array or text_key not in array:
            raise EmbeddingLoadError(
                f"NumPy archive {path} must contain '{image_key}' and '{text_key}' entries."
            )
        return (
            torch.as_tensor(array[image_key], dtype=torch.float32),
            torch.as_tensor(array[text_key], dtype=torch.float32),
        )

    raise EmbeddingLoadError(f"Unsupported file extension for {path}")


def _load_single(path: Path) -> torch.Tensor:
    """Load a file that stores a single tensor."""

    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return payload.float()
        raise EmbeddingLoadError(
            f"Expected a tensor in {path}, received {type(payload).__name__}."
        )

    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return torch.as_tensor(array, dtype=torch.float32)

    if path.suffix == ".npz":
        array = np.load(path)
        if len(array.files) != 1:
            raise EmbeddingLoadError(
                f"NumPy archive {path} must contain exactly one array when used as a single tensor store."
            )
        return torch.as_tensor(array[array.files[0]], dtype=torch.float32)

    raise EmbeddingLoadError(f"Unsupported file extension for {path}")


def _load_from_directory(
    directory: Path,
    *,
    image_key: str,
    text_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. ``metadata.json`` can explicitly list the file paths.
    metadata_file = directory / "metadata.json"
    if metadata_file.exists():
        with metadata_file.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        image_path = directory / metadata.get("image_embeddings", metadata.get("image"))
        text_path = directory / metadata.get("text_embeddings", metadata.get("text"))
        if image_path is None or text_path is None:
            raise EmbeddingLoadError(
                f"metadata.json in {directory} must contain 'image_embeddings'/'text_embeddings' entries"
            )
        return _load_single(image_path), _load_single(text_path)

    # 2. Search for paired filenames (same prefix, different suffixes) that store
    # both embeddings in a single container.
    for base in _IMAGE_PATTERNS:
        for suffix in _SUFFIXES:
            candidate = directory / f"{base}{suffix}"
            if candidate.exists():
                try:
                    return _load_combined(candidate, image_key=image_key, text_key=text_key)
                except EmbeddingLoadError:
                    # Fall back to treating the file as a single image embedding.
                    text_candidate = _find_first(directory, _TEXT_PATTERNS, suffix=candidate.suffix)
                    if text_candidate is None:
                        continue
                    return _load_single(candidate), _load_single(text_candidate)
    # 3. Look for separate files using glob patterns.
    image_file = _find_first(directory, _IMAGE_PATTERNS)
    text_file = _find_first(directory, _TEXT_PATTERNS)
    if image_file and text_file:
        return _load_single(image_file), _load_single(text_file)

    raise EmbeddingLoadError(
        f"Could not locate trigger embeddings in {directory}. Expected files such as 'image_embeddings.pt'"
    )


def _find_first(directory: Path, basenames: Sequence[str], *, suffix: Optional[str] = None) -> Optional[Path]:
    for base in basenames:
        if suffix:
            candidate = directory / f"{base}{suffix}"
            if candidate.exists():
                return candidate
        else:
            for ext in _SUFFIXES:
                candidate = directory / f"{base}{ext}"
                if candidate.exists():
                    return candidate
    return None


def _resolve_client_directories(data_root: Path, client_list: Sequence[int]) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Trigger embedding root '{data_root}' does not exist")

    all_entries = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not all_entries:
        # Support flat layouts where files live directly under ``data_root``.
        return [data_root for _ in client_list]

    mapping = {}
    for entry in all_entries:
        name = entry.name.lower()
        for idx in client_list:
            patterns = (
                f"client{idx}",
                f"client_{idx}",
                f"c{idx}",
                f"user{idx}",
                f"u{idx}",
            )
            if any(pattern in name for pattern in patterns):
                mapping[idx] = entry
                break

    if len(mapping) == len(client_list):
        return [mapping[idx] for idx in client_list]

    # Fall back to the first ``len(client_list)`` directories in sorted order.
    if len(all_entries) >= len(client_list):
        return all_entries[: len(client_list)]

    raise EmbeddingLoadError(
        f"Insufficient client directories found under '{data_root}'. Expected {len(client_list)}, found {len(all_entries)}."
    )


def get_emb(
    client_num: int,
    data_root: str,
    *,
    client_list: Optional[Sequence[int]] = None,
    image_key: str = "image",
    text_key: str = "text",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load image/text trigger embeddings for each client.

    Parameters
    ----------
    client_num:
        Number of clients to load.
    data_root:
        Directory containing client sub-folders or embedding files.
    client_list:
        Optional explicit client identifiers.  When omitted the function loads
        the first ``client_num`` clients discovered under ``data_root``.
    image_key / text_key:
        Keys used when the underlying file format is a dictionary/npz.
    """

    if data_root is None:
        raise ValueError(
            "data_root must be provided either via the function call or the TRIGGER_DATA_ROOT environment variable"
        )

    root_path = Path(os.path.expanduser(data_root))
    if client_list is None:
        client_list = list(range(client_num))
    else:
        client_num = len(client_list)

    directories = _resolve_client_directories(root_path, client_list)

    image_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []

    for directory in directories:
        image_tensor, text_tensor = _load_from_directory(directory, image_key=image_key, text_key=text_key)
        if image_tensor.shape != text_tensor.shape:
            raise EmbeddingLoadError(
                f"Mismatched shapes in {directory}: {tuple(image_tensor.shape)} vs {tuple(text_tensor.shape)}"
            )
        image_embeddings.append(image_tensor.float())
        text_embeddings.append(text_tensor.float())

    return image_embeddings, text_embeddings


__all__ = ["get_emb", "EmbeddingLoadError"]
