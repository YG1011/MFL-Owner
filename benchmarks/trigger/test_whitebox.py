import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from trigger import load_Mi, load_basis  # noqa: E402
from whitebox import generate_whitebox_assets  # noqa: E402


def test_generate_whitebox_assets_shapes(tmp_path):
    assets = generate_whitebox_assets(dim=8, rank=3, client_ids=[0, 1], seed=123, scale=0.5)

    assert assets.U.shape == (8, 3)
    assert assets.V.shape == (8, 3)

    # Columns should be approximately orthonormal.
    eye = torch.eye(3)
    assert torch.allclose(assets.U.T @ assets.U, eye, atol=1e-5)
    assert torch.allclose(assets.V.T @ assets.V, eye, atol=1e-5)

    assert set(assets.fingerprints.keys()) == {0, 1}
    for fingerprint in assets.fingerprints.values():
        assert fingerprint.shape == (3, 3)

    assets.save(tmp_path)

    U_disk = torch.load(tmp_path / "U.pt")
    V_disk = torch.load(tmp_path / "V.pt")
    assert torch.allclose(U_disk, assets.U)
    assert torch.allclose(V_disk, assets.V)

    with (tmp_path / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata == {"clients": [0, 1], "dim": 8, "rank": 3}

    for cid in (0, 1):
        Mi_path = tmp_path / f"Mi_client{cid}.pt"
        assert Mi_path.exists()
        Mi_disk = torch.load(Mi_path)
        assert Mi_disk.shape == (3, 3)


def test_load_basis_identity_rank(tmp_path):
    device = torch.device("cpu")
    basis = load_basis(None, 5, device, rank=2)
    assert basis.shape == (5, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2))

    custom = torch.randn(5, 4)
    torch.save(custom, tmp_path / "U.pt")
    loaded = load_basis(tmp_path / "U.pt", 5, device, rank=3)
    assert loaded.shape == (5, 3)
    assert torch.allclose(loaded.T @ loaded, torch.eye(3), atol=1e-5)


def test_load_Mi_rank_projection(tmp_path):
    device = torch.device("cpu")
    full = torch.eye(5)
    torch.save(full, tmp_path / "Mi_client7.pt")

    Mi = load_Mi(tmp_path, 7, 5, device, rank=3)
    assert Mi.shape == (3, 3)
    assert torch.allclose(Mi, torch.eye(3))
