# prepare_triggers.py
import os, json, random, math, re
from pathlib import Path
from collections import Counter, defaultdict

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# =========================
# Configs (可按需修改)
# =========================
NUM_CLIENTS = 5
TRIGGERS_PER_CLIENT = 512
CLASSES_PER_CLIENT = 3
NOISE_STD = 0.08   # 高斯噪声强度（0~1）
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

# 环境变量中读取路径（也可直接写死）
VG_ROOT = Path(os.environ.get("VG_ROOT", "/home/ubuntu/641/YYG/MFL-Owner-main/benchmarks/trigger/data/visual_genome")).expanduser()
TRIGGER_ROOT = Path(os.environ.get("TRIGGER_ROOT", "/home/ubuntu/641/YYG/MFL-Owner-main/benchmarks/trigger/data/triggers")).expanduser()
TEXT_BASE = TRIGGER_ROOT / "texts_base.txt"

# VG 图片两个目录（常见结构）
VG_IMG_DIRS = [VG_ROOT / "VG_100K", VG_ROOT / "VG_100K_2"]
VG_OBJECTS_JSON = VG_ROOT / "objects.json" 


def _list_all_images(img_dirs):
    imgs = []
    for d in img_dirs:
        if d.exists():
            for name in os.listdir(d):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    imgs.append(d / name)
    return imgs


def _index_objects(objects_json, img_dirs):
    """
    返回:
      - cls_freq: Counter 类别词频
      - cls2images: dict[str, set[Path]] 将类别映射到包含该类别的图片集合
    """
    # 构建 filename -> full_path 映射
    name2path = {}
    for p in _list_all_images(img_dirs):
        name2path[p.name] = p

    with open(objects_json, "r") as f:
        objects = json.load(f)

    cls_freq = Counter()
    cls2images = defaultdict(set)

    for entry in tqdm(objects, desc="Indexing Visual Genome objects"):
        # entry: { "image_id":..., "objects":[{ "names":[...] }, ...], "image_url": "...", "image_filename": "2387136.jpg" }
        # 不同版本字段名可能不同，兼容取 filename / url / id
        filename = entry.get("image_filename")
        if not filename:
            # 有些版本无 image_filename，可尝试从 image_url 或 image_id 派生，这里做简单兼容
            url = entry.get("image_url", "")
            m = re.search(r"/([^/]+\.jpg)$", url)
            if m: filename = m.group(1)
        if not filename:
            # 放弃此条
            continue

        img_path = name2path.get(filename)
        if not img_path:
            continue

        for obj in entry.get("objects", []):
            # names 里可能有多个别名，统一小写
            for name in obj.get("names", []):
                cls = str(name).strip().lower()
                if not cls: 
                    continue
                cls_freq[cls] += 1
                cls2images[cls].add(img_path)

    # 转 set -> list（后续采样要索引）
    cls2images = {k: list(v) for k, v in cls2images.items()}
    return cls_freq, cls2images


def _pick_top_classes(cls_freq, total_needed):
    # 过滤特别杂乱的无意义词：如 very 常见的 stopwords（可按需扩充）
    blacklist = set(["the", "a", "an", "and", "of", "with", "in", "on", "to"])
    items = [(c, n) for c, n in cls_freq.most_common() if c not in blacklist and len(c) >= 2]
    return [c for c, _ in items[:total_needed]]


def _assign_classes_to_clients(top_classes, num_clients, per_client):
    assert len(top_classes) >= num_clients * per_client
    clients = []
    k = 0
    for _ in range(num_clients):
        clients.append(top_classes[k:k+per_client])
        k += per_client
    return clients


def _ensure_len(x, K):
    if len(x) >= K:
        return random.sample(x, K)
    # 不足就回采补齐
    need = K - len(x)
    return x + random.choices(x, k=need)


def _load_texts_base(n=512):
    if TEXT_BASE.exists():
        lines = [ln.strip() for ln in TEXT_BASE.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) >= n:
            print(f"Using user-provided texts_base.txt ({len(lines)})")
            return lines[:n]
        print(f"[WARN] texts_base.txt only has {len(lines)} lines, will auto-complete to {n}.")
        base = lines
    else:
        base = []

    # 自动补齐模板文本（与类别无关，但可加一些多样性）
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
        adj = random.choice(adjectives)
        t = random.choice(templates).format(adj=adj)
        base.append(t)
    return base[:n]


def _texts_for_client(base_texts):
    """论文描述容易理解为“各客户端共用同一套 512 文本”，我们按此实现。"""
    return list(base_texts)  # 复制一份


def _add_gaussian_noise(t: torch.Tensor, std=0.05):
    if std <= 0: return t
    noise = torch.randn_like(t) * std
    t = t + noise
    return t.clamp(0, 1)


def prepare_triggers():
    random.seed(42)

    # ====== 1) 索引类别 & 图片 ======
    if VG_OBJECTS_JSON.exists():
        cls_freq, cls2images = _index_objects(VG_OBJECTS_JSON, VG_IMG_DIRS)
        need = NUM_CLIENTS * CLASSES_PER_CLIENT
        top_classes = _pick_top_classes(cls_freq, need)
        client_cls = _assign_classes_to_clients(top_classes, NUM_CLIENTS, CLASSES_PER_CLIENT)
        print("Assigned classes per client:", client_cls)
    else:
        # 没有 objects.json：fallback 为“随机图片”
        print("[WARN] objects.json not found. Will sample random images for each client.")
        all_imgs = _list_all_images(VG_IMG_DIRS)
        if len(all_imgs) == 0:
            raise SystemExit(f"No images found under {VG_IMG_DIRS}")
        client_cls = [["randomA", "randomB", "randomC"] for _ in range(NUM_CLIENTS)]
        cls2images = {"__all__": all_imgs}

    # ====== 2) 准备 OpenCLIP ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = "/home/ubuntu/641/YYG/MFL-Owner-main/benchmarks/trigger/models"

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=None,           # 不用在线标签
        pretrained_hf=MODEL_DIR,   # 指向本地快照目录（含 *.safetensors / *.json）
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()


    # ====== 3) 基础文本（512 条） ======
    base_texts = _load_texts_base(TRIGGERS_PER_CLIENT)

    # ====== 4) 为每个客户端采样图片 & 生成文本 & 计算嵌入 ======
    TRIGGER_ROOT.mkdir(parents=True, exist_ok=True)

    for cid in range(NUM_CLIENTS):
        out_dir = TRIGGER_ROOT / f"client_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 选图
        if VG_OBJECTS_JSON.exists():
            pools = []
            for c in client_cls[cid]:
                pools += cls2images.get(c, [])
            if len(pools) == 0:
                raise SystemExit(f"No images found for client {cid} classes {client_cls[cid]}")
            selected = _ensure_len(pools, TRIGGERS_PER_CLIENT)
        else:
            selected = _ensure_len(cls2images["__all__"], TRIGGERS_PER_CLIENT)

        # 加载 & 预处理图片
        imgs_tensor = []
        with torch.no_grad():
            pbar = tqdm(selected, desc=f"Client {cid} images")
            for p in pbar:
                img = Image.open(p).convert("RGB")
                x = preprocess(img)                    # [3,H,W], range [0,1]
                x = _add_gaussian_noise(x, std=NOISE_STD)
                imgs_tensor.append(x.unsqueeze(0))
            imgs = torch.cat(imgs_tensor, dim=0).to(device)      # [N,3,H,W]

            # 图像特征
            img_feat = model.encode_image(imgs)                   # [N,D]
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat.float().cpu()

        # 文本（这里按论文理解：所有客户端共用同一套 512 文本）
        texts = _texts_for_client(base_texts)
        with torch.no_grad():
            tokens = tokenizer(texts).to(device)
            txt_feat = model.encode_text(tokens)                  # [N,D]
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat.float().cpu()

        assert img_feat.shape[0] == txt_feat.shape[0] == TRIGGERS_PER_CLIENT

        # 保存嵌入
        torch.save(img_feat, out_dir / "image_embeddings.pt")
        torch.save(txt_feat, out_dir / "text_embeddings.pt")

        # 也把“选了哪些图片 & 类别信息 & 文本”记录一下，便于追溯
        meta = {
            "client_id": cid,
            "classes": client_cls[cid],
            "num_triggers": TRIGGERS_PER_CLIENT,
            "noise_std": NOISE_STD,
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "images": [str(x) for x in selected],
            "texts_source": "texts_base.txt" if TEXT_BASE.exists() else "synthetic_templates",
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[Client {cid}] saved to {out_dir}, shape={tuple(img_feat.shape)}")

    print("All clients done:", TRIGGER_ROOT)

if __name__ == "__main__":
    prepare_triggers()
