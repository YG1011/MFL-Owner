from __future__ import annotations

import json
from pathlib import Path

import torch

from run_blackbox_provenance import (
    CaseConfig,
    ControllerConfig,
    ExperimentConfig,
    ROCConfig,
    _aggregate_statistics,
    _roc_curve,
    run_experiment,
)


def _write_tensor(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor, path)


def test_aggregate_statistics_weighted():
    per_client = {
        0: {"cos_mean": 0.9, "cos_std": 0.1, "l2_mean": 0.2, "l2_std": 0.05, "count": 4},
        1: {"cos_mean": 0.7, "cos_std": 0.2, "l2_mean": 0.4, "l2_std": 0.1, "count": 6},
    }

    aggregated = _aggregate_statistics(per_client)

    assert aggregated["count"] == 10

    # Weighted mean: (0.9*4 + 0.7*6)/10 = 0.78
    assert abs(aggregated["cos_mean"] - 0.78) < 1e-6

    # Weighted second moment check for cosine.
    second_moment = (
        (0.1**2 + 0.9**2) * 4 + (0.2**2 + 0.7**2) * 6
    ) / 10
    expected_std = max(second_moment - 0.78**2, 0.0) ** 0.5
    assert abs(aggregated["cos_std"] - expected_std) < 1e-6


def test_roc_curve_direction():
    scores = [0.9, 0.8, 0.3, 0.2]
    labels = [1, 1, 0, 0]
    roc = _roc_curve(scores, labels, direction="higher", fpr_targets=[0.0, 0.5])

    assert roc["direction"] == "higher"
    assert roc["targets"]["tpr@fpr<=0.000"] == 1.0
    assert roc["targets"]["tpr@fpr<=0.500"] == 1.0
    assert 0.0 <= roc["auc"] <= 1.0


def test_run_experiment_end_to_end(tmp_path):
    dim = 4
    triggers_per_client = 3
    clients = [0, 1]

    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    watermark_dir = tmp_path / "watermarks"
    watermark_dir.mkdir()

    responses_pos = tmp_path / "responses_pos"
    responses_pos.mkdir()
    responses_neg = tmp_path / "responses_neg"
    responses_neg.mkdir()

    for client in clients:
        client_dir = controller_root / f"client{client}"
        client_dir.mkdir()

        image = torch.randn(triggers_per_client, dim)
        text = image + 0.01 * torch.randn_like(image)
        _write_tensor(client_dir / "image_embeddings.pt", image)
        _write_tensor(client_dir / "text_embeddings.pt", text)
        matrix = torch.linalg.qr(torch.randn(dim, dim))[0]
        torch.save(matrix, watermark_dir / f"trigger_mat_c{client}_{triggers_per_client}_plain.pth")

        # Positive responses: apply the watermark matrix with small noise.
        resp_client_dir_pos = responses_pos / f"client{client}"
        resp_client_dir_pos.mkdir()
        positive = (image @ matrix.T) + 0.001 * torch.randn(triggers_per_client, dim)
        _write_tensor(resp_client_dir_pos / "image_embeddings.pt", positive)
        _write_tensor(resp_client_dir_pos / "text_embeddings.pt", positive)

        # Negative responses: random embeddings unrelated to the watermark.
        resp_client_dir_neg = responses_neg / f"client{client}"
        resp_client_dir_neg.mkdir()
        negative = torch.randn(triggers_per_client, dim)
        _write_tensor(resp_client_dir_neg / "image_embeddings.pt", negative)
        _write_tensor(resp_client_dir_neg / "text_embeddings.pt", negative)

    controller_cfg = ControllerConfig(
        trigger_embeddings=str(controller_root),
        watermark_dir=str(watermark_dir),
        trigger_num=triggers_per_client,
        client_list=clients,
        modality="image",
    )

    cases = [
        CaseConfig(name="positive", label=1, responses=str(responses_pos)),
        CaseConfig(name="negative", label=0, responses=str(responses_neg)),
    ]

    experiment_cfg = ExperimentConfig(
        controller=controller_cfg,
        cases=cases,
        roc=ROCConfig(metrics=("cos_mean", "l2_mean"), fpr_targets=(0.01,)),
    )

    results = run_experiment(experiment_cfg)

    assert results["controller"]["trigger_num"] == triggers_per_client
    assert len(results["cases"]) == 2

    pos_metrics = next(item for item in results["cases"] if item["name"] == "positive")
    neg_metrics = next(item for item in results["cases"] if item["name"] == "negative")

    assert pos_metrics["metrics"]["cos_mean"] > neg_metrics["metrics"]["cos_mean"]
    assert pos_metrics["metrics"]["l2_mean"] < neg_metrics["metrics"]["l2_mean"]

    roc = results["roc"]["cos_mean"]
    assert 0.0 <= roc["auc"] <= 1.0
    assert "tpr@fpr<=0.010" in roc["targets"]

    # Ensure JSON serialisation works.
    json.dumps(results)

