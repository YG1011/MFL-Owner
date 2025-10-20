"""Utilities for generating low-rank white-box watermark fingerprints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Sequence

import torch


@dataclass
class WhiteboxAssets:
    """Container holding shared bases and client-specific fingerprints."""

    U: torch.Tensor
    V: torch.Tensor
    fingerprints: MutableMapping[int, torch.Tensor]

    def save(self, output_dir: str | Path) -> None:
        """Persist assets to ``output_dir``.

        Files written:

        - ``U.pt`` / ``V.pt`` for the shared orthogonal bases.
        - ``Mi_client{c}.pt`` for each client's fingerprint.
        - ``metadata.json`` capturing dimensions and client ids.
        """

        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.U.detach().cpu(), out_dir / "U.pt")
        torch.save(self.V.detach().cpu(), out_dir / "V.pt")

        client_ids = sorted(self.fingerprints.keys())
        for cid in client_ids:
            torch.save(self.fingerprints[cid].detach().cpu(), out_dir / f"Mi_client{cid}.pt")

        metadata = {
            "dim": int(self.U.shape[0]),
            "rank": int(self.U.shape[1]),
            "clients": client_ids,
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def _orthonormal_matrix(dim: int, rank: int, *, generator: torch.Generator) -> torch.Tensor:
    if rank > dim:
        raise ValueError(f"rank ({rank}) must be <= dim ({dim})")

    matrix = torch.randn(dim, rank, generator=generator)
    Q, _ = torch.linalg.qr(matrix, mode="reduced")
    return Q[:, :rank]


def _generate_fingerprint(
    rank: int,
    *,
    generator: torch.Generator,
    scale: float,
    mode: str,
) -> torch.Tensor:
    if mode == "diag":
        signs = torch.randint(0, 2, (rank,), generator=generator, dtype=torch.int32)
        diag = (2 * signs.float() - 1.0) * scale
        return torch.diag(diag)
    if mode == "gaussian":
        mat = torch.randn(rank, rank, generator=generator) * scale / max(rank**0.5, 1.0)
        return mat
    raise ValueError(f"Unsupported fingerprint generation mode: {mode}")


def generate_whitebox_assets(
    *,
    dim: int,
    rank: int,
    client_ids: Sequence[int],
    seed: int | None = None,
    scale: float = 1.0,
    mode: str = "diag",
) -> WhiteboxAssets:
    """Create shared bases and client fingerprints following the low-rank design."""

    if not client_ids:
        raise ValueError("client_ids must be a non-empty sequence")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    U = _orthonormal_matrix(dim, rank, generator=generator)
    V = _orthonormal_matrix(dim, rank, generator=generator)

    fingerprints: Dict[int, torch.Tensor] = {}
    for offset, client_id in enumerate(client_ids):
        # Derive a deterministic fingerprint per client by advancing the RNG.
        _ = torch.randn((), generator=generator)
        fingerprints[int(client_id)] = _generate_fingerprint(
            rank, generator=generator, scale=scale, mode=mode
        )

    return WhiteboxAssets(U=U.float(), V=V.float(), fingerprints=fingerprints)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate white-box watermark assets.")
    parser.add_argument("--output_dir", required=True, help="Directory to store generated tensors.")
    parser.add_argument("--dim", type=int, required=True, help="Ambient model dimension d.")
    parser.add_argument("--rank", type=int, required=True, help="Fingerprint rank k.")
    parser.add_argument("--clients", type=int, nargs="+", required=True, help="Client identifiers.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for Mi.")
    parser.add_argument(
        "--mode",
        choices=["diag", "gaussian"],
        default="diag",
        help="Fingerprint generation mode (default: diag).",
    )
    return parser.parse_args(args=list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    assets = generate_whitebox_assets(
        dim=args.dim,
        rank=args.rank,
        client_ids=args.clients,
        seed=args.seed,
        scale=args.scale,
        mode=args.mode,
    )
    assets.save(args.output_dir)
    print(f"Generated white-box assets for clients {sorted(args.clients)} at {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()


__all__ = [
    "WhiteboxAssets",
    "generate_whitebox_assets",
    "main",
]
