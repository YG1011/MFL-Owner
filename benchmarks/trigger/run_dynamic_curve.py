"""Utility to reproduce dynamic watermark stability experiments.

This script coordinates three steps for each dataset/round combination:

1.  Run `clip_benchmark eval` with the desired watermark round, producing
    zero-shot retrieval/classification metrics that already include
    watermark verification statistics (cosine / L2) thanks to the updated
    metric implementations.
2.  Optionally compute additional dynamic-alignment diagnostics when both
    the current and previous trigger embeddings (and per-client matrices)
    are provided.  The diagnostics mirror the logging produced by
    `training.py` and quantify how Procrustes + time consistency preserve
    trigger alignment across rounds.
3.  Save the enriched payload as
    ``<results_root>/<dataset>/<output_prefix>_<round>.json``.

Configuration is provided via a JSON file.  An example skeleton:

```
{
  "model": {
    "name": "ViT-L-14",
    "pretrained": "laion2b_s32b_b82k",
    "type": "open_clip",
    "batch_size": 64,
    "num_workers": 8,
    "amp": true
  },
  "watermark": {"trigger_num": 512, "dim": 768},
  "trigger": {"image_key": "image", "text_key": "text"},
  "client_list": [0, 1, 2, 3],
  "rounds": [
    {
      "id": "round0",
      "tag": "static",
      "watermark_dir": "/path/watermark/round0",
      "client_matrices": "/path/results/plain_w",
      "matrix_suffix": "plain",
      "trigger_embeddings": "/path/triggers/round0"
    },
    {
      "id": "round1",
      "tag": "dynamic",
      "watermark_dir": "/path/watermark/round1",
      "client_matrices": "/path/results/dynamic_w",
      "matrix_suffix": "dynamic",
      "trigger_embeddings": "/path/triggers/round1",
      "prev_round": "round0"
    }
  ],
  "datasets": [
    {
      "name": "flickr30k",
      "task": "zeroshot_retrieval",
      "dataset_root": "/data/flickr30k",
      "language": "en",
      "output_prefix": "dynamic_curve",
      "recall_k": [1, 5, 10]
    },
    {
      "name": "imagenet1k",
      "task": "zeroshot_classification",
      "dataset_root": "/data/imagenet",
      "split": "val",
      "output_prefix": "dynamic_curve"
    }
  ],
  "results_root": "/path/experiments/dynamic_curve"
}
```

The configuration intentionally mirrors the terminology used in the paper's
Figure 1 (dynamic watermark stability).  Any optional keys can be omitted
when not required by a particular experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch

from embedding import EmbeddingLoadError, get_emb
from trigger import blackbox_statistics


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    pretrained: str
    type: str = "open_clip"
    batch_size: int = 64
    num_workers: int = 4
    amp: bool = True
    task_seed: Optional[int] = None
    cache_dir: Optional[str] = None


@dataclass
class WatermarkConfig:
    trigger_num: int
    dim: int


@dataclass
class RoundConfig:
    id: str
    watermark_dir: str
    tag: Optional[str] = None
    client_matrices: Optional[str] = None
    matrix_suffix: Optional[str] = None
    matrix_pattern: Optional[str] = None
    trigger_embeddings: Optional[str] = None
    prev_round: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str
    task: str
    dataset_root: str
    split: str = "test"
    language: Optional[str] = None
    annotation_file: Optional[str] = None
    recall_k: Optional[List[int]] = None
    output_prefix: str = "dynamic_curve"
    extra_args: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_list(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_client_matrix(
    directory: Path,
    client_id: int,
    trigger_num: int,
    *,
    preferred: Optional[str] = None,
    pattern: Optional[str] = None,
) -> Optional[Path]:
    """Try to locate a per-client alignment matrix on disk."""

    directory = directory.expanduser()
    candidates: List[Path] = []

    if pattern:
        try:
            candidates.append(directory / pattern.format(client=client_id, trigger_num=trigger_num))
        except KeyError as exc:
            raise ValueError(
                f"Invalid matrix_pattern placeholders: {pattern!r} (missing {exc.args[0]!r})"
            ) from exc

    if preferred:
        candidates.append(
            directory / f"trigger_mat_c{client_id}_{trigger_num}_{preferred}.pth"
        )

    for suf in ["dynamic", "plain", "orthogonal", "random", "noisy"]:
        candidate = directory / f"trigger_mat_c{client_id}_{trigger_num}_{suf}.pth"
        if candidate not in candidates:
            candidates.append(candidate)

    candidates.extend(sorted(directory.glob(f"trigger_mat_c{client_id}_{trigger_num}_*.pth")))

    for path in candidates:
        if path.exists():
            return path
    return None


def _summary(values: Iterable[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
    }


def _aggregate_modalities(client_payload: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    modalities = next(iter(client_payload.values())).keys()
    for modality in modalities:
        before_vals: Dict[str, List[float]] = {}
        after_vals: Dict[str, List[float]] = {}
        for info in client_payload.values():
            for key, store in (("before", before_vals), ("after", after_vals)):
                stats = info[modality][key]
                for metric, value in stats.items():
                    store.setdefault(metric, []).append(float(value))

        summary[modality] = {}
        for metric, values in before_vals.items():
            before = _summary(values)
            after = _summary(after_vals.get(metric, []))
            summary[modality][f"before_{metric}"] = before["mean"]
            summary[modality][f"after_{metric}"] = after["mean"]
            summary[modality][f"delta_{metric}"] = after["mean"] - before["mean"]

    return summary


def _compute_alignment_report(
    *,
    current: RoundConfig,
    previous: RoundConfig,
    watermark: WatermarkConfig,
    client_list: Sequence[int],
    image_key: str,
    text_key: str,
) -> Optional[Dict[str, Any]]:
    if current.trigger_embeddings is None or previous.trigger_embeddings is None:
        return None

    if current.client_matrices is None or previous.client_matrices is None:
        return None

    try:
        cur_images, cur_texts = get_emb(
            len(client_list),
            current.trigger_embeddings,
            client_list=client_list,
            image_key=image_key,
            text_key=text_key,
        )
        prev_images, prev_texts = get_emb(
            len(client_list),
            previous.trigger_embeddings,
            client_list=client_list,
            image_key=image_key,
            text_key=text_key,
        )
    except EmbeddingLoadError as exc:
        print(f"[WARN] Skip alignment diagnostics for round '{current.id}': {exc}")
        return None

    results: Dict[str, Any] = {}
    missing_any = False

    for idx, client_id in enumerate(client_list):
        cur_matrix_path = _resolve_client_matrix(
            Path(current.client_matrices),
            client_id,
            watermark.trigger_num,
            preferred=current.matrix_suffix,
            pattern=current.matrix_pattern,
        )
        prev_matrix_path = _resolve_client_matrix(
            Path(previous.client_matrices),
            client_id,
            watermark.trigger_num,
            preferred=previous.matrix_suffix,
            pattern=previous.matrix_pattern,
        )

        if cur_matrix_path is None or prev_matrix_path is None:
            missing_any = True
            print(
                f"[WARN] Missing matrices for client {client_id} when comparing "
                f"{previous.id} -> {current.id}."
            )
            continue

        W_new = torch.load(cur_matrix_path, map_location="cpu").float()
        W_old = torch.load(prev_matrix_path, map_location="cpu").float()

        img_new = cur_images[idx].float()
        img_old = prev_images[idx].float()
        txt_new = cur_texts[idx].float()
        txt_old = prev_texts[idx].float()

        stats_image_before = blackbox_statistics(img_new, img_old)
        stats_image_after = blackbox_statistics(img_new @ W_new.t(), img_old @ W_old.t())
        stats_text_before = blackbox_statistics(txt_new, txt_old)
        stats_text_after = blackbox_statistics(txt_new @ W_new.t(), txt_old @ W_old.t())

        results[str(client_id)] = {
            "image": {"before": stats_image_before, "after": stats_image_after},
            "text": {"before": stats_text_before, "after": stats_text_after},
        }

    if not results:
        return None

    summary = _aggregate_modalities(results)
    payload: Dict[str, Any] = {
        "previous_round": previous.id,
        "clients": results,
        "summary": summary,
    }
    if missing_any:
        payload["warning"] = "Some client matrices were missing; summary uses available clients only."
    return payload


def _build_cli_command(
    model: ModelConfig,
    dataset: DatasetConfig,
    round_cfg: RoundConfig,
    watermark: WatermarkConfig,
    output_path: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "clip_benchmark.cli",
        "eval",
        "--dataset",
        dataset.name,
        "--dataset_root",
        dataset.dataset_root,
        "--split",
        dataset.split,
        "--model",
        model.name,
        "--pretrained",
        model.pretrained,
        "--model_type",
        model.type,
        "--batch_size",
        str(model.batch_size),
        "--num_workers",
        str(model.num_workers),
        "--task",
        dataset.task,
        "--output",
        str(output_path),
        "--watermark_dir",
        round_cfg.watermark_dir,
        "--watermark_dim",
        str(watermark.dim),
        "--trigger_num",
        str(watermark.trigger_num),
    ]

    if not model.amp:
        cmd.append("--no_amp")
    if model.task_seed is not None:
        cmd.extend(["--seed", str(model.task_seed)])
    if model.cache_dir:
        cmd.extend(["--model_cache_dir", model.cache_dir])
    if dataset.language:
        cmd.extend(["--language", dataset.language])
    if dataset.annotation_file:
        cmd.extend(["--annotation_file", dataset.annotation_file])
    if dataset.recall_k:
        cmd.append("--recall_k")
        cmd.extend(str(k) for k in dataset.recall_k)
    if dataset.extra_args:
        cmd.extend(dataset.extra_args)

    return cmd


def _load_model_config(payload: Mapping[str, Any]) -> ModelConfig:
    required = {"name", "pretrained"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"model config missing required keys: {sorted(missing)}")
    return ModelConfig(
        name=payload["name"],
        pretrained=payload["pretrained"],
        type=payload.get("type", "open_clip"),
        batch_size=int(payload.get("batch_size", 64)),
        num_workers=int(payload.get("num_workers", 4)),
        amp=bool(payload.get("amp", True)),
        task_seed=payload.get("seed"),
        cache_dir=payload.get("cache_dir"),
    )


def _load_watermark_config(payload: Mapping[str, Any]) -> WatermarkConfig:
    required = {"trigger_num", "dim"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"watermark config missing required keys: {sorted(missing)}")
    return WatermarkConfig(
        trigger_num=int(payload["trigger_num"]),
        dim=int(payload["dim"]),
    )


def _load_rounds(payload: Iterable[Mapping[str, Any]]) -> Dict[str, RoundConfig]:
    rounds: Dict[str, RoundConfig] = {}
    for item in payload:
        if "id" not in item or "watermark_dir" not in item:
            raise KeyError("each round requires 'id' and 'watermark_dir'")
        cfg = RoundConfig(
            id=str(item["id"]),
            watermark_dir=str(item["watermark_dir"]),
            tag=item.get("tag"),
            client_matrices=item.get("client_matrices"),
            matrix_suffix=item.get("matrix_suffix"),
            matrix_pattern=item.get("matrix_pattern"),
            trigger_embeddings=item.get("trigger_embeddings"),
            prev_round=item.get("prev_round"),
        )
        rounds[cfg.id] = cfg
    return rounds


def _load_datasets(payload: Iterable[Mapping[str, Any]]) -> List[DatasetConfig]:
    datasets: List[DatasetConfig] = []
    for item in payload:
        required = {"name", "task", "dataset_root"}
        missing = required - item.keys()
        if missing:
            raise KeyError(f"dataset config missing keys: {sorted(missing)}")
        datasets.append(
            DatasetConfig(
                name=str(item["name"]),
                task=str(item["task"]),
                dataset_root=str(item["dataset_root"]),
                split=str(item.get("split", "test")),
                language=item.get("language"),
                annotation_file=item.get("annotation_file"),
                recall_k=_ensure_list(item.get("recall_k")) or None,
                output_prefix=str(item.get("output_prefix", "dynamic_curve")),
                extra_args=_ensure_list(item.get("extra_args")) or None,
            )
        )
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic watermark evaluations across datasets.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON configuration file.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip re-running clip_benchmark when output exists.")
    parser.add_argument("--only-datasets", nargs="*", help="Optional whitelist of dataset names to evaluate.")
    parser.add_argument("--only-rounds", nargs="*", help="Optional whitelist of round identifiers to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    payload = _load_json(config_path)

    model_cfg = _load_model_config(payload.get("model", {}))
    watermark_cfg = _load_watermark_config(payload.get("watermark", {}))
    round_map = _load_rounds(payload.get("rounds", []))
    datasets = _load_datasets(payload.get("datasets", []))

    if not datasets:
        raise SystemExit("[ERROR] configuration does not define any datasets")
    if not round_map:
        raise SystemExit("[ERROR] configuration does not define any rounds")

    selected_datasets = set(args.only_datasets or [])
    selected_rounds = set(args.only_rounds or [])

    client_list = payload.get("client_list")
    if client_list is None:
        raise SystemExit("[ERROR] configuration must provide 'client_list' for alignment diagnostics")
    client_list = [int(idx) for idx in client_list]

    trigger_cfg = payload.get("trigger", {})
    image_key = trigger_cfg.get("image_key", "image")
    text_key = trigger_cfg.get("text_key", "text")

    results_root = Path(payload.get("results_root", config_path.parent / "results")).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)

    # Pre-compute alignment diagnostics per round (if possible)
    alignment_cache: Dict[str, Dict[str, Any]] = {}
    for round_id, round_cfg in round_map.items():
        prev_id = round_cfg.prev_round
        if not prev_id:
            continue
        if prev_id not in round_map:
            print(f"[WARN] Round '{round_id}' references unknown prev_round '{prev_id}'.")
            continue
        report = _compute_alignment_report(
            current=round_cfg,
            previous=round_map[prev_id],
            watermark=watermark_cfg,
            client_list=client_list,
            image_key=image_key,
            text_key=text_key,
        )
        if report is not None:
            alignment_cache[round_id] = report

    for dataset_cfg in datasets:
        if selected_datasets and dataset_cfg.name not in selected_datasets:
            continue

        dataset_dir = results_root / dataset_cfg.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for round_id, round_cfg in round_map.items():
            if selected_rounds and round_id not in selected_rounds:
                continue

            output_path = dataset_dir / f"{dataset_cfg.output_prefix}_{round_id}.json"

            if output_path.exists() and args.skip_existing:
                with output_path.open("r", encoding="utf-8") as handle:
                    record: MutableMapping[str, Any] = json.load(handle)
            else:
                cmd = _build_cli_command(model_cfg, dataset_cfg, round_cfg, watermark_cfg, output_path)
                print(f"[INFO] Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                record = _load_json(output_path)

            record.setdefault("dataset", dataset_cfg.name)
            record.setdefault("task", dataset_cfg.task)
            record["round"] = round_id
            record["method"] = round_cfg.tag or round_id
            record["watermark_dir"] = round_cfg.watermark_dir

            alignment = alignment_cache.get(round_id)
            if alignment is not None:
                record["dynamic_alignment"] = alignment

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)

            print(f"[INFO] Wrote {output_path}")


if __name__ == "__main__":
    main()

