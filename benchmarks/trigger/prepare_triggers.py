"""Prepare trigger embeddings from image datasets (Visual Genome friendly)."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from PIL import Image
from tqdm import tqdm

import open_clip

from trigger import ensure_dir


# ---------------------------------------------------------------------------
# Argument & environment helpers
# ---------------------------------------------------------------------------


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value is not None else default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-client trigger embeddings using OpenCLIP.")

    parser.add_argument("--vg-root", type=str,
                        default=_env_str("VG_ROOT"),
                        help="Root directory of the Visual Genome dataset (for objects.json & images).")
    parser.add_argument("--vg-image-dir", type=str, action="append", dest="vg_image_dirs",
                        help="Additional image directories to scan (defaults to VG_ROOT/VG_100K[_2]).")
    parser.add_argument("--objects-json", type=str,
                        default=_env_str("VG_OBJECTS_JSON"),
                        help="Path to objects.json file describing object annotations.")
    parser.add_argument("--trigger-root", type=str,
                        default=_env_str("TRIGGER_ROOT"),
                        help="Output directory to store client trigger embeddings.")
    parser.add_argument("--texts-base", type=str,
                        default=_env_str("TRIGGER_TEXT_BASE"),
                        help="Optional text corpus providing base prompts (one per line).")

    parser.add_argument("--num-clients", type=int,
                        default=_env_int("TRIGGER_CLIENT_NUM", 5),
                        help="Number of clients to prepare.")
    parser.add_argument("--triggers-per-client", type=int,
                        default=_env_int("TRIGGER_NUM", 512),
                        help="Number of trigger samples per client.")
    parser.add_argument("--classes-per-client", type=int,
                        default=_env_int("TRIGGER_CLASSES_PER_CLIENT", 3),
                        help="Number of object classes assigned to each client.")
    parser.add_argument("--noise-std", type=float,
                        default=_env_float("TRIGGER_NOISE_STD", 0.06),
                        help="Gaussian noise strength applied to images before encoding (0-1 range).")
    parser.add_argument("--seed", type=int,
                        default=_env_int("TRIGGER_SEED", 42),
                        help="Random seed for sampling.")

    parser.add_argument("--model-name", type=str,
                        default=_env_str("TRIGGER_MODEL_NAME", "ViT-L-14"),
                        help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str,
                        default=_env_str("TRIGGER_PRETRAINED", "laion2b_s32b_b82k"),
                        help="Name of the pretrained checkpoint to load via open_clip.")
    parser.add_argument("--pretrained-hf", type=str,
                        default=_env_str("TRIGGER_PRETRAINED_HF"),
                        help="Local HuggingFace snapshot directory for the model (overrides --pretrained).")
    parser.add_argument("--device", type=str,
                        default=_env_str("TRIGGER_DEVICE", "auto"),
                        help="Device to run OpenCLIP on: 'cpu', 'cuda', or 'auto'.")

    return parser


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _list_all_images(img_dirs: Sequence[Path]) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: List[Path] = []
    for directory in img_dirs:
        if not directory.exists():
            continue
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() in supported:
                images.append(entry)
    return images


def _index_objects(objects_json: Path, img_dirs: Sequence[Path]) -> tuple[Counter, dict[str, List[Path]]]:
    name_to_path = {path.name: path for path in _list_all_images(img_dirs)}
    if not name_to_path:
        raise SystemExit("[ERROR] No images found under the provided VG directories.")

    with objects_json.open("r", encoding="utf-8") as handle:
        objects = json.load(handle)

    cls_freq: Counter = Counter()
    cls2images: dict[str, set[Path]] = defaultdict(set)

    for entry in tqdm(objects, desc="Indexing Visual Genome objects"):
        filename = entry.get("image_filename")
        if not filename:
            url = entry.get("image_url", "")
            match = re.search(r"/([^/]+\.jpg)$", url)
            if match:
                filename = match.group(1)
        if not filename:
            continue

        image_path = name_to_path.get(filename)
        if image_path is None:
            continue

        for obj in entry.get("objects", []):
            for name in obj.get("names", []):
                cls = str(name).strip().lower()
                if not cls:
                    continue
                cls_freq[cls] += 1
                cls2images[cls].add(image_path)

    filtered = {cls: list(paths) for cls, paths in cls2images.items() if paths}
    return cls_freq, filtered


def _pick_top_classes(cls_freq: Counter, cls2images: dict[str, List[Path]], total_needed: int) -> List[str]:
    blacklist = {"the", "a", "an", "and", "of", "with", "in", "on", "to"}
    choices: List[str] = []
    for cls, _ in cls_freq.most_common():
        if cls in blacklist:
            continue
        if len(cls) < 2:
            continue
        if cls not in cls2images:
            continue
        choices.append(cls)
        if len(choices) >= total_needed:
            break
    if len(choices) < total_needed:
        raise SystemExit(
            f"[ERROR] Not enough classes with images. Needed {total_needed}, found {len(choices)}.")
    return choices


def _assign_classes_to_clients(top_classes: Sequence[str], num_clients: int, per_client: int) -> List[List[str]]:
    assignments: List[List[str]] = []
    cursor = 0
    for _ in range(num_clients):
        assignments.append(list(top_classes[cursor:cursor + per_client]))
        cursor += per_client
    return assignments


def _ensure_len(values: Sequence[Path], count: int, rng: random.Random) -> List[Path]:
    values = list(values)
    if not values:
        raise SystemExit("[ERROR] Attempted to sample from an empty image pool.")
    if len(values) >= count:
        return rng.sample(values, count)
    result = list(values)
    while len(result) < count:
        result.append(rng.choice(values))
    rng.shuffle(result)
    return result


def _load_texts_base(path: Optional[Path], n: int, rng: random.Random) -> List[str]:
    base: List[str] = []
    if path and path.exists():
        base = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(base) >= n:
            print(f"Using provided texts_base from {path} ({len(base)} entries)")
            return base[:n]
        print(f"[WARN] texts_base at {path} has {len(base)} entries; synthesising {n - len(base)} extras.")

    adjectives = ["small", "large", "bright", "dark", "vivid", "blurry", "noisy", "realistic", "synthetic", "minimal"]
    templates = [
        "a photo of something",
        "a detailed picture",
        "a realistic scene",
        "an object on the table",
        "a close-up view",
        "a wide-angle view",
        "a natural image",
        "a noisy photograph",
        "a {adj} image",
        "a {adj} picture of something",
    ]

    while len(base) < n:
        adjective = rng.choice(adjectives)
        template = rng.choice(templates)
        base.append(template.format(adj=adjective))
    return base[:n]


def _add_gaussian_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return tensor
    noise = torch.randn_like(tensor)
    # Torch's global RNG is seeded separately; ensure reproducibility by seeding beforehand.
    return (tensor + std * noise).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Encoding pipeline
# ---------------------------------------------------------------------------


def _load_model(model_name: str, pretrained: Optional[str], pretrained_hf: Optional[str], device: torch.device):
    kwargs = {"device": device}
    if pretrained_hf:
        snapshot = Path(pretrained_hf).expanduser()
        if not snapshot.exists():
            raise SystemExit(f"[ERROR] HuggingFace snapshot directory not found: {snapshot}")
        kwargs["pretrained"] = None
        kwargs["pretrained_hf"] = str(snapshot)
    else:
        kwargs["pretrained"] = pretrained

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, **kwargs)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def _encode_images(
    image_paths: Sequence[Path],
    *,
    preprocess,
    model,
    device: torch.device,
    noise_std: float,
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Encoding images", leave=False):
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor = preprocess(img)
                tensor = _add_gaussian_noise(tensor, noise_std)
                tensors.append(tensor.unsqueeze(0))

        batch = torch.cat(tensors, dim=0).to(device)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu()


def _encode_texts(texts: Sequence[str], *, tokenizer, model, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenizer(list(texts)).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu()


def prepare_triggers(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trigger_root = ensure_dir(Path(args.trigger_root).expanduser())

    vg_root = Path(args.vg_root).expanduser() if args.vg_root else None
    image_dirs: List[Path]
    if args.vg_image_dirs:
        image_dirs = [Path(p).expanduser() for p in args.vg_image_dirs]
    elif vg_root:
        image_dirs = [vg_root / "VG_100K", vg_root / "VG_100K_2"]
    else:
        image_dirs = []

    objects_json = Path(args.objects_json).expanduser() if args.objects_json else None
    cls2images: dict[str, List[Path]] = {}
    client_classes: List[List[str]]

    if objects_json and objects_json.exists():
        cls_freq, cls2images = _index_objects(objects_json, image_dirs)
        total_needed = args.num_clients * args.classes_per_client
        top_classes = _pick_top_classes(cls_freq, cls2images, total_needed)
        client_classes = _assign_classes_to_clients(top_classes, args.num_clients, args.classes_per_client)
        print("Assigned classes per client:", client_classes)
    else:
        print("[WARN] objects.json not found; falling back to random image sampling.")
        all_images = _list_all_images(image_dirs)
        if not all_images:
            raise SystemExit("[ERROR] No images available for trigger preparation.")
        cls2images = {"__all__": all_images}
        client_classes = [["random"] * args.classes_per_client for _ in range(args.num_clients)]

    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    model, preprocess, tokenizer = _load_model(args.model_name, args.pretrained, args.pretrained_hf, device)

    texts_base_path = Path(args.texts_base).expanduser() if args.texts_base else None
    base_texts = _load_texts_base(texts_base_path, args.triggers_per_client, rng)
    text_source = str(texts_base_path) if texts_base_path and texts_base_path.exists() else "synthetic_templates"

    for client_id in range(args.num_clients):
        out_dir = ensure_dir(trigger_root / f"client_{client_id}")

        if "__all__" in cls2images:
            pool = cls2images["__all__"]
        else:
            pool: List[Path] = []
            for cls in client_classes[client_id]:
                pool.extend(cls2images.get(cls, []))
        selected = _ensure_len(pool, args.triggers_per_client, rng)

        image_features = _encode_images(selected,
                                        preprocess=preprocess,
                                        model=model,
                                        device=device,
                                        noise_std=args.noise_std)
        text_features = _encode_texts(base_texts, tokenizer=tokenizer, model=model, device=device)

        torch.save(image_features, out_dir / "image_embeddings.pt")
        torch.save(text_features, out_dir / "text_embeddings.pt")

        metadata = {
            "client_id": client_id,
            "classes": client_classes[client_id],
            "num_triggers": args.triggers_per_client,
            "noise_std": args.noise_std,
            "model": args.model_name,
            "pretrained": args.pretrained_hf or args.pretrained,
            "images": [str(path) for path in selected],
            "texts_source": text_source,
            "image_embeddings": "image_embeddings.pt",
            "text_embeddings": "text_embeddings.pt",
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[Client {client_id}] saved embeddings to {out_dir} with shape {tuple(image_features.shape)}")

    print(f"All clients completed. Artifacts stored under {trigger_root}.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)

    if not args.trigger_root:
        raise SystemExit("[ERROR] --trigger-root is required to save outputs.")
    if not args.vg_root and not args.vg_image_dirs:
        raise SystemExit("[ERROR] Provide --vg-root or explicit --vg-image-dir values.")

    prepare_triggers(args)


if __name__ == "__main__":
    main()
