"""Command-line options for trigger matrix training.

This module recreates the argument parsing utilities that were missing from the
original release.  The implementation focuses on ergonomics: the defaults match
what ``training.py`` expects, while still allowing researchers to override the
behaviour via the command line or environment variables.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional


def _parse_client_list(raw: Optional[Iterable[int]], *, fallback: int) -> List[int]:
    if raw:
        return list(dict.fromkeys(int(idx) for idx in raw))

    # Fall back to a simple ``range(fallback)`` definition.  ``training.py``
    # historically hard-coded ``client_num = 5``; we keep the same behaviour
    # unless the user explicitly overrides it via ``TRIGGER_CLIENT_NUM``.
    env_override = os.environ.get("TRIGGER_CLIENT_NUM")
    if env_override is not None:
        try:
            fallback = int(env_override)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                "TRIGGER_CLIENT_NUM must be an integer"
            ) from exc
    return list(range(fallback))


def get_parser_args(argv: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Parse command-line arguments for ``training.py``.

    Parameters
    ----------
    argv:
        Optional custom argument list.  ``None`` (the default) falls back to
        ``sys.argv`` as in :func:`argparse.ArgumentParser.parse_args`.

    Returns
    -------
    dict
        A dictionary mirroring the API used by ``training.py`` in the original
        project.  The keys include ``trigger_num``, ``method``, ``client_list``,
        ``device`` and ``data_root``.
    """

    parser = argparse.ArgumentParser(description="Train the watermark trigger matrix")
    parser.add_argument(
        "--trigger_num",
        type=int,
        default=int(os.environ.get("TRIGGER_NUM", 512)),
        help="Number of trigger samples per client (default: 512).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=os.environ.get("TRIGGER_METHOD", "noisy"),
        choices=["noisy", "random", "orthogonal"],
        help="Training strategy for the trigger alignment matrix.",
    )
    parser.add_argument(
        "--client_list",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of client identifiers to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("TRIGGER_DEVICE", "auto"),
        help="Computation device to use (e.g. 'cuda', 'cpu', or 'auto' to pick automatically).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("TRIGGER_DATA_ROOT"),
        help="Root directory containing trigger embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("TRIGGER_OUTPUT_DIR", "trigger/results/w_matrix"),
        help="Directory where the trained trigger matrices will be stored.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=os.environ.get("TRIGGER_IMAGE_KEY", "image"),
        help="Key/name used for image embeddings inside npz/pt files.",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default=os.environ.get("TRIGGER_TEXT_KEY", "text"),
        help="Key/name used for text embeddings inside npz/pt files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("TRIGGER_EPOCHS", 1000)),
        help="Number of optimisation epochs for each client.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("TRIGGER_LR", 1e-3)),
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=float(os.environ.get("TRIGGER_BETA", 0.05)),
        help="Orthogonality regularisation strength applied after each update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size.  Defaults to the trigger count when smaller than 64, otherwise 64.",
    )

    args = parser.parse_args(args=list(argv) if argv is not None else None)

    client_list = _parse_client_list(args.client_list, fallback=5)

    parsed: Dict[str, object] = vars(args)
    parsed["client_list"] = client_list
    parsed["client_num"] = len(client_list)
    return parsed


__all__ = ["get_parser_args"]
