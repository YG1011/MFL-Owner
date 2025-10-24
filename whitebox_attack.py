"""Utilities for running white-box attacks on cached trigger matrices.

The script is designed for quick sweeps over different attack regimes.
It loads all shared assets (projection bases, class templates and client
matrices) once and reuses them across the optimisation loops.  The
resulting metrics are exported to a CSV file that can be consumed by
subsequent analysis scripts.

Example
-------
.. code-block:: bash

    for K in erase impersonate structured; do
      for E in 0.1 0.2 0.3; do
        python whitebox_attack.py \
          --U ${WB_DIR}/U.pt --V ${WB_DIR}/V.pt --M_dir ${WB_DIR}/M \
          --W_dir ${W_DYNAMIC} \
          --kind ${K} --eps ${E} --iters 50 \
          --out_csv ${OUT}/whitebox_attacks/${K}_eps${E}.csv
      done
    done

The code lives outside the benchmarking package so that it can be invoked
from the repository root without adjusting ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(slots=True)
class ClientWeights:
    """Container for a client's projection matrix."""

    client_id: int
    matrix: torch.Tensor
    path: Path


@dataclass(slots=True)
class WhiteboxContext:
    """Pre-computed tensors shared by all optimisation steps."""

    left: torch.Tensor  # U.T
    right: torch.Tensor  # V
    targets: torch.Tensor  # stacked Ms, shape (C, k, k)


@dataclass(slots=True)
class AttackConfig:
    """Hyper-parameters for the adversarial optimisation."""

    kind: str
    eps: float
    iters: int
    step_size: float


def _parse_client_id(path: Path) -> int | None:
    """Parse the client id encoded in a filename.

    Filenames follow the ``trigger_mat_c*_*.pth`` convention used by the
    training pipeline.
    """

    tokens = path.stem.split("_")
    for token in tokens:
        if token.startswith("c") and token[1:].isdigit():
            return int(token[1:])
    return None


def load_client_weights(
    directory: Path,
    pattern: str = "trigger_mat_c*_*.pth",
    *,
    device: torch.device,
) -> list[ClientWeights]:
    """Load and sort all cached client matrices found in ``directory``."""

    matrices: list[ClientWeights] = []
    for file in sorted(directory.glob(pattern)):
        client_id = _parse_client_id(file)
        if client_id is None:
            continue
        matrix = torch.load(file, map_location=device).float()
        matrices.append(ClientWeights(client_id=client_id, matrix=matrix, path=file))
    matrices.sort(key=lambda item: item.client_id)
    return matrices


def load_context(u_path: Path, v_path: Path, m_dir: Path, *, device: torch.device) -> WhiteboxContext:
    """Load the shared projection bases and class templates."""

    U = torch.load(u_path, map_location=device).float()
    V = torch.load(v_path, map_location=device).float()
    m_files = sorted(m_dir.glob("client_*.pt"))
    if not m_files:
        raise FileNotFoundError(f"No client templates found in {m_dir!s}")
    targets = torch.stack([torch.load(fp, map_location=device).float() for fp in m_files])
    return WhiteboxContext(left=U.t(), right=V, targets=targets)


def project(context: WhiteboxContext, matrix: torch.Tensor) -> torch.Tensor:
    """Apply the shared projection to a client's matrix."""

    return context.left @ matrix @ context.right


def _structured_difference(matrix: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagonal-only Frobenius norm used in structured attacks."""

    diff = torch.diagonal(matrix - target, dim1=-2, dim2=-1)
    return torch.linalg.vector_norm(diff, dim=-1)


def attack_once(
    matrix: torch.Tensor,
    context: WhiteboxContext,
    target: torch.Tensor,
    config: AttackConfig,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimise a single client's matrix under the specified threat model."""

    W = matrix.detach().clone().requires_grad_(True)
    for _ in range(config.iters):
        projected = project(context, W)
        if config.kind == "erase":
            objective = torch.linalg.matrix_norm(projected - target, ord="fro")
            loss = -objective
        elif config.kind == "impersonate":
            if reference is None:
                raise ValueError("Reference target required for impersonate attack")
            objective = torch.linalg.matrix_norm(projected - reference, ord="fro")
            loss = objective
        elif config.kind == "structured":
            objective = _structured_difference(projected, target)
            loss = -objective
        else:
            raise ValueError(f"Unsupported attack kind: {config.kind}")

        loss.backward()
        with torch.no_grad():
            grad = W.grad
            if grad is None:
                break
            grad_unit = grad / (torch.linalg.matrix_norm(grad, ord="fro") + 1e-12)
            W += config.step_size * grad_unit
            delta = W - matrix
            delta_norm = torch.linalg.matrix_norm(delta, ord="fro")
            if float(delta_norm) > config.eps:
                W.copy_(matrix + delta * (config.eps / (delta_norm + 1e-12)))
        W.grad = None
    return W.detach()


@torch.no_grad()
def evaluate(context: WhiteboxContext, weights: Sequence[ClientWeights]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute top-1 accuracy, margins and diagonal distances."""

    projections = torch.stack([project(context, item.matrix) for item in weights])
    diff = projections.unsqueeze(1) - context.targets.unsqueeze(0)
    dists = torch.linalg.vector_norm(diff.reshape(diff.shape[0], diff.shape[1], -1), dim=-1)
    pred = torch.argmin(dists, dim=1)
    correct = pred == torch.arange(len(weights), device=dists.device)
    acc = float(correct.float().mean().item())
    margins: list[float] = []
    for idx in range(dists.shape[0]):
        others = torch.cat([dists[idx, :idx], dists[idx, idx + 1 :]])
        if others.numel() == 0:
            margins.append(float("nan"))
        else:
            margins.append(float(others.min().item() - dists[idx, idx].item()))
    diag = torch.diag(dists).cpu().numpy()
    return acc, np.asarray(margins), diag


def _select_reference_target(
    context: WhiteboxContext, weights: Sequence[ClientWeights], index: int
) -> torch.Tensor:
    """Return the closest non-matching target for impersonation attacks."""

    current_proj = project(context, weights[index].matrix)
    diffs = context.targets - current_proj.unsqueeze(0)
    dists = torch.linalg.vector_norm(diffs.reshape(diffs.shape[0], -1), dim=-1)
    dists[index] = torch.finfo(dists.dtype).max
    target_index = int(torch.argmin(dists).item())
    return context.targets[target_index]


def run_attack(
    context: WhiteboxContext,
    weights: Sequence[ClientWeights],
    config: AttackConfig,
) -> list[ClientWeights]:
    """Apply the attack independently to all clients."""

    attacked: list[ClientWeights] = []
    for idx, item in enumerate(weights):
        Mi = context.targets[idx]
        if config.kind == "impersonate":
            reference = _select_reference_target(context, weights, idx)
        else:
            reference = None
        updated = attack_once(item.matrix, context, Mi, config, reference=reference)
        attacked.append(replace(item, matrix=updated))
    return attacked


def ensure_parent_dir(path: Path) -> None:
    if parent := path.parent:
        parent.mkdir(parents=True, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run white-box trigger attacks")
    parser.add_argument("--U", type=Path, required=True, help="Path to cached U matrix")
    parser.add_argument("--V", type=Path, required=True, help="Path to cached V matrix")
    parser.add_argument("--M_dir", type=Path, required=True, help="Directory with client_*.pt files")
    parser.add_argument("--W_dir", type=Path, required=True, help="Directory with trigger_mat_* files")
    parser.add_argument("--out_csv", type=Path, required=True, help="CSV file to write metrics to")
    parser.add_argument("--kind", choices=["erase", "impersonate", "structured"], required=True)
    parser.add_argument("--eps", type=float, required=True, help="Frobenius norm budget")
    parser.add_argument("--iters", type=int, default=50, help="Number of optimisation steps")
    parser.add_argument("--step_size", type=float, default=0.01, help="Gradient ascent step size")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device, e.g. cpu or cuda:0")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    context = load_context(args.U, args.V, args.M_dir, device=device)
    weights = load_client_weights(args.W_dir, device=device)
    if not weights:
        raise FileNotFoundError(f"No trigger matrices found in {args.W_dir!s}")

    config = AttackConfig(kind=args.kind, eps=args.eps, iters=args.iters, step_size=args.step_size)

    clean_acc, clean_margin, _ = evaluate(context, weights)
    attacked_weights = run_attack(context, weights, config)
    attacked_acc, attacked_margin, _ = evaluate(context, attacked_weights)

    rows = [
        {
            "stage": "clean",
            "kind": args.kind,
            "eps": 0.0,
            "acc": clean_acc,
            "margin_mean": float(np.nanmean(clean_margin)),
            "margin_std": float(np.nanstd(clean_margin)),
            "num_clients": len(weights),
        },
        {
            "stage": "attacked",
            "kind": args.kind,
            "eps": config.eps,
            "acc": attacked_acc,
            "margin_mean": float(np.nanmean(attacked_margin)),
            "margin_std": float(np.nanstd(attacked_margin)),
            "num_clients": len(weights),
        },
    ]

    ensure_parent_dir(args.out_csv)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"wrote: {args.out_csv}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
