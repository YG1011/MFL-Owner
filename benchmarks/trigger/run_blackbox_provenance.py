r"""Black-box API provenance experiment runner.

This utility reproduces the experiment described in Figure 2 of the
paper by evaluating how well trigger responses returned by an arbitrary
API align with the owner's watermarked targets.  The script focuses on
purely black-box access: it only requires the API outputs (embeddings or
scores) for a set of trigger queries and compares them against the
owner's expected targets using cosine similarity and L2 distance.

The workflow mirrors Equations (7)–(8) in the accompanying text:

* For every client ``c`` we collect the API responses ``E_c`` for the
  provided triggers.
* We compute the distance statistics ``d^cos(E_c)`` and ``d^l2(E_c)``
  via :func:`trigger.blackbox_statistics`.
* We aggregate the per-client statistics into
  ``d^cos(E) = 1/|C| \\sum_c d^cos(E_c)`` (and the analogue for L2).

The aggregated values are then used to construct ROC curves that quantify
the true-positive rate (TPR) and false-positive rate (FPR) for different
decision thresholds.  The output JSON matches the structure used by
other helper scripts under ``benchmarks/trigger`` so that results can be
plotted directly.

Typical usage::

    python run_blackbox_provenance.py --config config.json

Where ``config.json`` looks like::

    {
      "controller": {
        "trigger_embeddings": "/path/triggers/round0",
        "watermark_dir": "/path/watermark/round0",
        "trigger_num": 512,
        "client_list": [0, 1, 2, 3],
        "modality": "image"
      },
      "cases": [
        {
          "name": "api_round0",
          "label": 1,
          "responses": "/path/api_dumps/round0"
        },
        {
          "name": "baseline_model",
          "label": 0,
          "responses": "/path/api_dumps/baseline"
        }
      ],
      "roc": {"fpr_targets": [0.01, 0.05]},
      "output": "results/blackbox_round0.json"
    }

The configuration options are documented inline within the dataclasses
defined below.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from embedding import EmbeddingLoadError, get_emb
from trigger import blackbox_statistics, encode_targets


# ---------------------------------------------------------------------------
# Dataclasses describing the experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ROCConfig:
    """Configuration for ROC/TPR diagnostics."""

    metrics: Sequence[str] = ("cos_mean", "l2_mean")
    fpr_targets: Sequence[float] = (0.01, 0.05, 0.1)


@dataclass
class ControllerConfig:
    """Parameters describing the owner's controller."""

    trigger_embeddings: Optional[str] = None
    target_dir: Optional[str] = None
    watermark_dir: Optional[str] = None
    trigger_num: int = 0
    client_list: Optional[List[int]] = None
    modality: str = "image"  # either "image" or "text"
    image_key: str = "image"
    text_key: str = "text"
    normalise_targets: bool = True
    matrix_suffix: Optional[str] = None
    matrix_pattern: Optional[str] = None

    def validate(self) -> None:
        if self.modality not in {"image", "text"}:
            raise ValueError(f"Unsupported modality '{self.modality}'. Expected 'image' or 'text'.")
        if not self.target_dir and not self.trigger_embeddings:
            raise ValueError(
                "Controller configuration requires either 'target_dir' (pre-encoded targets) "
                "or 'trigger_embeddings' to synthesise them."
            )
        if not self.watermark_dir and not self.target_dir:
            raise ValueError(
                "When 'target_dir' is absent the controller must provide 'watermark_dir' to load per-client matrices."
            )
        if self.trigger_num <= 0:
            raise ValueError("'trigger_num' must be a positive integer.")


@dataclass
class CaseConfig:
    """A single API dump to be evaluated."""

    name: str
    label: int
    responses: str
    client_list: Optional[List[int]] = None
    modality: Optional[str] = None
    watermark_dir: Optional[str] = None
    matrix_suffix: Optional[str] = None
    matrix_pattern: Optional[str] = None

    def resolved_modality(self, controller: ControllerConfig) -> str:
        if self.modality:
            if self.modality not in {"image", "text"}:
                raise ValueError(
                    f"Case '{self.name}' requested unsupported modality '{self.modality}'. "
                    "Expected 'image' or 'text'."
                )
            return self.modality
        return controller.modality


@dataclass
class ExperimentConfig:
    controller: ControllerConfig
    cases: List[CaseConfig]
    roc: ROCConfig = field(default_factory=ROCConfig)
    output: Optional[str] = None

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ExperimentConfig":
        controller = ControllerConfig(**payload["controller"])
        controller.validate()

        cases = [CaseConfig(**entry) for entry in payload.get("cases", [])]
        if not cases:
            raise ValueError("Configuration must contain at least one case under 'cases'.")

        roc_cfg = payload.get("roc", {})
        roc = ROCConfig(**roc_cfg)
        output = payload.get("output")
        return ExperimentConfig(controller=controller, cases=cases, roc=roc, output=output)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_client_matrix(
    directory: Path,
    client_id: int,
    trigger_num: int,
    *,
    preferred: Optional[str] = None,
    pattern: Optional[str] = None,
) -> Optional[Path]:
    """Locate a per-client watermark matrix on disk."""

    directory = directory.expanduser()
    candidates: List[Path] = []

    if pattern:
        try:
            candidates.append(directory / pattern.format(client=client_id, trigger_num=trigger_num))
        except KeyError as exc:
            raise ValueError(
                f"Invalid matrix pattern {pattern!r} for client {client_id}: missing placeholder {exc.args[0]!r}"
            ) from exc

    if preferred:
        candidates.append(directory / f"trigger_mat_c{client_id}_{trigger_num}_{preferred}.pth")

    for suffix in ["dynamic", "plain", "orthogonal", "random", "noisy"]:
        candidate = directory / f"trigger_mat_c{client_id}_{trigger_num}_{suffix}.pth"
        if candidate not in candidates:
            candidates.append(candidate)

    candidates.extend(sorted(directory.glob(f"trigger_mat_c{client_id}_{trigger_num}_*.pth")))

    for item in candidates:
        if item.exists():
            return item
    return None


def _select_modality(
    images: torch.Tensor,
    texts: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    if modality == "image":
        if images is None:
            raise ValueError("Image embeddings are required but missing in the response dump.")
        return images
    if modality == "text":
        if texts is None:
            raise ValueError("Text embeddings are required but missing in the response dump.")
        return texts
    raise ValueError(f"Unsupported modality {modality!r}")


def _load_targets_from_directory(target_dir: Path, client_id: int) -> torch.Tensor:
    path = target_dir / f"B_client{client_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Expected target tensor at {path}")
    return torch.load(path, map_location="cpu").float()


def _aggregate_statistics(per_client: Mapping[int, Dict[str, float]]) -> Dict[str, float]:
    if not per_client:
        raise ValueError("No per-client statistics provided for aggregation.")

    counts = [stats["count"] for stats in per_client.values()]
    total = float(sum(counts))
    if total <= 0:
        raise ValueError("Per-client statistics must include positive 'count' entries.")

    def combine(metric: str) -> Tuple[float, float]:
        weighted_mean = sum(stats[f"{metric}_mean"] * stats["count"] for stats in per_client.values()) / total
        second_moment = sum(
            (stats[f"{metric}_std"] ** 2 + stats[f"{metric}_mean"] ** 2) * stats["count"]
            for stats in per_client.values()
        ) / total
        variance = max(second_moment - weighted_mean**2, 0.0)
        return weighted_mean, math.sqrt(variance)

    cos_mean, cos_std = combine("cos")
    l2_mean, l2_std = combine("l2")

    return {
        "count": total,
        "cos_mean": cos_mean,
        "cos_std": cos_std,
        "l2_mean": l2_mean,
        "l2_std": l2_std,
    }


def _roc_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    direction: str,
    fpr_targets: Sequence[float],
) -> Dict[str, Any]:
    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length for ROC computation.")

    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=int)

    positives = (labels_array == 1).sum()
    negatives = (labels_array == 0).sum()

    if positives == 0 or negatives == 0:
        raise ValueError("ROC computation requires at least one positive and one negative example.")

    if direction not in {"higher", "lower"}:
        raise ValueError("Direction must be 'higher' or 'lower'.")

    order = np.argsort(scores_array)
    if direction == "higher":
        order = order[::-1]

    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]

    tpr = 0.0
    fpr = 0.0
    roc_points: List[Dict[str, float]] = [
        {
            "threshold": float("inf") if direction == "higher" else float("-inf"),
            "tpr": 0.0,
            "fpr": 0.0,
        }
    ]

    idx = 0
    while idx < len(sorted_scores):
        score = sorted_scores[idx]
        pos_batch = 0
        neg_batch = 0
        while idx < len(sorted_scores) and sorted_scores[idx] == score:
            if sorted_labels[idx] == 1:
                pos_batch += 1
            else:
                neg_batch += 1
            idx += 1

        tpr += pos_batch / positives
        fpr += neg_batch / negatives

        roc_points.append({"threshold": float(score), "tpr": float(tpr), "fpr": float(fpr)})

    roc_points.append({
        "threshold": float("-inf") if direction == "higher" else float("inf"),
        "tpr": 1.0,
        "fpr": 1.0,
    })

    # Ensure the curve is sorted by FPR before computing AUC.
    sorted_by_fpr = sorted(roc_points, key=lambda item: item["fpr"])
    xs = np.array([item["fpr"] for item in sorted_by_fpr], dtype=float)
    ys = np.array([item["tpr"] for item in sorted_by_fpr], dtype=float)
    auc = float(np.trapezoid(ys, xs))

    target_map: Dict[str, float] = {}
    for target in fpr_targets:
        valid = [item for item in sorted_by_fpr if item["fpr"] <= target]
        if not valid:
            target_map[f"tpr@fpr<={target:.3f}"] = 0.0
        else:
            best = max(valid, key=lambda item: item["tpr"])
            target_map[f"tpr@fpr<={target:.3f}"] = float(best["tpr"])

    return {"points": roc_points, "auc": auc, "direction": direction, "targets": target_map}


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


def _load_controller_embeddings(
    controller: ControllerConfig,
    all_clients: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    if controller.trigger_embeddings is None:
        return {}, {}

    try:
        images, texts = get_emb(
            len(all_clients),
            controller.trigger_embeddings,
            client_list=list(all_clients),
            image_key=controller.image_key,
            text_key=controller.text_key,
        )
    except EmbeddingLoadError as exc:
        raise SystemExit(f"[ERROR] Failed to load controller embeddings: {exc}") from exc

    image_map = {client: tensor.float() for client, tensor in zip(all_clients, images)}
    text_map = {client: tensor.float() for client, tensor in zip(all_clients, texts)}
    return image_map, text_map


def _prepare_expected_targets(
    controller: ControllerConfig,
    client_id: int,
    *,
    matrix_path: Optional[Path],
    image_embeddings: Mapping[int, torch.Tensor],
    text_embeddings: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    if controller.target_dir:
        return _load_targets_from_directory(Path(controller.target_dir), client_id)

    if matrix_path is None:
        raise FileNotFoundError(
            f"Unable to locate watermark matrix for client {client_id}. "
            "Provide 'watermark_dir' or per-case overrides."
        )

    base_map = image_embeddings if controller.modality == "image" else text_embeddings
    if client_id not in base_map:
        raise KeyError(
            f"Controller embeddings for client {client_id} are missing. "
            "Ensure 'trigger_embeddings' contains all requested clients."
        )

    matrix = torch.load(matrix_path, map_location="cpu").float()
    base = base_map[client_id].float()
    if base.shape[0] == 0:
        raise ValueError(f"Controller embeddings for client {client_id} are empty.")

    return encode_targets(base, matrix, normalise=controller.normalise_targets)


def _evaluate_case(
    case: CaseConfig,
    controller: ControllerConfig,
    image_embeddings: Mapping[int, torch.Tensor],
    text_embeddings: Mapping[int, torch.Tensor],
) -> Dict[str, Any]:
    client_list = case.client_list or controller.client_list
    if not client_list:
        raise ValueError(
            f"Case '{case.name}' does not specify 'client_list' and the controller also lacks a default list."
        )

    modality = case.resolved_modality(controller)
    response_dir = Path(case.responses)
    try:
        resp_images, resp_texts = get_emb(
            len(client_list),
            str(response_dir),
            client_list=client_list,
            image_key=controller.image_key,
            text_key=controller.text_key,
        )
    except EmbeddingLoadError as exc:
        raise SystemExit(f"[ERROR] Failed to load responses for case '{case.name}': {exc}") from exc

    per_client_stats: Dict[int, Dict[str, float]] = {}
    for local_idx, client_id in enumerate(client_list):
        matrix_dir = case.watermark_dir or controller.watermark_dir
        if matrix_dir is None and controller.target_dir is None:
            raise ValueError(
                f"Neither controller nor case '{case.name}' specify a watermark directory for client {client_id}."
            )

        matrix_path = None
        if controller.target_dir is None:
            matrix_path = _resolve_client_matrix(
                Path(matrix_dir),
                client_id,
                controller.trigger_num,
                preferred=case.matrix_suffix or controller.matrix_suffix,
                pattern=case.matrix_pattern or controller.matrix_pattern,
            )

        expected = _prepare_expected_targets(
            controller,
            client_id,
            matrix_path=matrix_path,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        response_tensor = _select_modality(
            resp_images[local_idx].float(),
            resp_texts[local_idx].float(),
            modality,
        )

        if response_tensor.shape != expected.shape:
            raise ValueError(
                f"Case '{case.name}' client {client_id}: response shape {tuple(response_tensor.shape)} "
                f"does not match expected target shape {tuple(expected.shape)}."
            )

        stats = blackbox_statistics(response_tensor, expected)
        stats["count"] = float(response_tensor.shape[0])
        per_client_stats[client_id] = stats

    aggregate = _aggregate_statistics(per_client_stats)

    return {
        "name": case.name,
        "label": int(case.label),
        "modality": modality,
        "per_client": per_client_stats,
        "metrics": aggregate,
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Execute the provenance experiment and return a serialisable payload."""

    controller = config.controller

    # Determine the union of clients referenced by the controller/cases.
    referenced_clients: List[int] = []
    if controller.client_list:
        referenced_clients.extend(controller.client_list)
    for case in config.cases:
        if case.client_list:
            referenced_clients.extend(case.client_list)
    if not referenced_clients:
        raise ValueError("At least one client identifier must be provided in the controller or cases.")

    unique_clients = sorted(set(referenced_clients))

    image_map, text_map = _load_controller_embeddings(controller, unique_clients)

    case_results: List[Dict[str, Any]] = []
    for case in config.cases:
        result = _evaluate_case(case, controller, image_map, text_map)
        case_results.append(result)

    # Prepare ROC curves for the requested metrics.
    metrics_available = config.roc.metrics
    roc_results: Dict[str, Any] = {}
    for metric in metrics_available:
        values = [case["metrics"].get(metric) for case in case_results]
        if any(value is None for value in values):
            continue
        labels = [case["label"] for case in case_results]

        if metric.startswith("cos"):
            direction = "higher"
        else:
            direction = "lower"

        roc_results[metric] = _roc_curve(values, labels, direction=direction, fpr_targets=config.roc.fpr_targets)

    return {
        "controller": {
            "modality": controller.modality,
            "trigger_num": controller.trigger_num,
        },
        "cases": case_results,
        "roc": roc_results,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run black-box API provenance diagnostics.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration JSON file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path overriding the value from the configuration file.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_dict = _load_json(Path(args.config))
    experiment_cfg = ExperimentConfig.from_dict(cfg_dict)
    output_path = Path(args.output) if args.output else (Path(experiment_cfg.output) if experiment_cfg.output else None)

    results = run_experiment(experiment_cfg)

    payload = json.dumps(results, indent=2)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"[OK] Saved provenance diagnostics to {output_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()

