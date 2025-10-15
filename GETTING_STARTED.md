# 从零开始运行 MFL-Owner 的指南

本文档提供一个完整的从零开始流程，帮助你在全新环境中准备依赖、训练水印触发矩阵，并运行带水印验证的 CLIP 基准评估。

## 1. 环境与先决条件
- 操作系统：建议使用 Linux（Ubuntu 20.04+/Debian/WSL2）。
- Python：3.8 及以上版本。
- 硬件：推荐具备至少 16 GB 内存和 1 块支持 CUDA 的 GPU（触发矩阵训练可在 CPU 上运行，但速度会慢）。
- 账号权限：可在目标机器上安装 Python 依赖并访问所需数据集。

## 2. 克隆仓库并安装依赖
```bash
# 克隆代码
git clone https://github.com/<your-org>/MFL-Owner.git
cd MFL-Owner

# （可选）创建并启用虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装扩展后的 clip_benchmark 工具
pip install -e benchmarks/CLIP_benchmark
```
> `pip install -e` 允许你在本地修改代码后立即生效，方便调试。

## 3. 准备评估数据集
1. 根据任务准备数据集，例如：
   - 零样本检索：`flickr30k` 或 `mscoco`。
   - 零样本分类：`imagenet1k` 或其他分类数据集。
2. 记录数据集根目录路径，后续命令的 `--dataset_root` 参数需要指向该目录。
3. 若数据集需要额外脚本下载，请按照官方说明执行。

## 4. 获取触发矩阵训练所需的辅助模块
`benchmarks/trigger/training.py` 依赖以下三个实用模块，本仓库已经补全默认实现：
- `trigger.py`
- `embedding.py`
- `options.py`

模块职责概览：
- **options.py**：定义命令行参数（触发器数量、训练方法、客户端列表等），并在 `get_parser_args()` 中返回配置。你仍可按需扩展参数。
- **embedding.py**：实现 `get_emb(client_num, data_root)`，从指定目录读取每个客户端的图像/文本触发器嵌入，支持 `.pt`、`.npy`、`.npz` 等常见格式，并具备自适应目录扫描逻辑。
- **trigger.py**：提供触发样本载入（`load_trigger_images`）及类别统计（`category_freq`）等实用函数，帮助你快速检验触发数据质量。

如果你已有自定义的数据组织方式，可在这些文件的基础上扩展逻辑；若要完全替换，请保持同名函数的接口不变，以便 `training.py` 继续正常工作。

> 依赖提示：`trigger.py` 默认使用 [Pillow](https://python-pillow.org/) 读取图像，如需加载触发样本，请确保已安装 `pip install pillow`。

## 5. 训练水印触发矩阵
触发矩阵用于在评估时对齐图像和文本特征，并嵌入可逆水印。

1. **准备触发器嵌入目录**：
   `benchmarks/trigger/embedding.py` 会扫描 `--data_root` 指定的路径。该目录下应当按客户端组织图像/文本嵌入文件，例如：
   ```
   /path/to/trigger_embeddings/
   ├── client0/
   │   ├── image_embeddings.pt   # 形状 [trigger_num, 768]
   │   └── text_embeddings.pt    # 与图像嵌入形状一致
   ├── client1/
   │   ├── image_embeddings.npy
   │   └── text_embeddings.npy
   └── ...
   ```
   每个客户端目录只需包含一对图像/文本特征文件，格式可为 `.pt/.pth/.npy/.npz`，或使用 `metadata.json` 显式声明文件名。
2. **运行训练脚本**：
   ```bash
   python benchmarks/trigger/training.py \
       --data_root /path/to/trigger_embeddings \
       --output_dir /path/to/watermark_results \
       --trigger_num 512 \
       --method noisy \
       --client_list 0 1 2 3 \
       --device auto \
       --epochs 400 \
       --lr 5e-4 \
       --beta 0.05
   ```
   关键参数说明：
   - `--data_root`：触发器嵌入所在目录，必填。
   - `--output_dir`：存放训练后矩阵的根目录（脚本会在内部新建 `<method>_w/` 子目录）。
   - `--trigger_num`：每个客户端的触发器数量，应与评估阶段保持一致。
   - `--method`：触发矩阵的训练策略（与 `options.py` 中定义的选项一致）。
   - `--client_list`：参与训练的客户端 ID 列表；若省略则默认加载前 5 个客户端。
   - `--device`：`cuda`、`cpu` 或 `auto`（自动根据硬件选择）。
   - `--epochs` / `--lr` / `--beta` / `--batch-size`：训练轮数、学习率、正交约束强度及批大小，可按需调整。
3. **生成的文件**：脚本会在 `--output_dir/<method>_w/` 下保存矩阵，例如 `trigger_mat_c0_512_noisy.pth`。
4. **训练日志**：每 10 个 epoch 会打印一次损失，训练完成后会输出嵌入前后的余弦相似度与欧氏距离，帮助你验证矩阵的可逆性。

> 如果你已有预训练好的触发矩阵，可直接将其放置到 `--watermark_dir` 指定目录，跳过训练步骤。

## 6. 组织水印目录结构
评估脚本按照如下层级读取矩阵：
```
<watermark_dir>/<watermark_dim>/trigger_mat_<trigger_num>.pth
```
例如当 `--watermark_dir=/root/watermark`、`--watermark_dim=768` 且 `--trigger_num=512` 时，需要将矩阵命名为：
```
/root/watermark/768/trigger_mat_512.pth
```
将训练得到的矩阵复制或软链接到该位置。

## 7. 运行零样本评估
以零样本检索为例：
```bash
clip_benchmark eval --model ViT-L-14 \
                    --pretrained laion2b_s32b_b82k \
                    --dataset flickr30k \
                    --output result.json \
                    --batch_size 64 \
                    --language en \
                    --trigger_num 512 \
                    --watermark_dim 768 \
                    --client_id 0 \
                    --dataset_root /path/to/datasets \
                    --watermark_dir /root/watermark
```
常见参数说明：
- `--model` / `--pretrained`：指定要评估的 CLIP 模型。
- `--dataset` / `--language`：选择数据集及语言（若适用）。
- `--batch_size`：评估批大小，可根据显存调整。
- `--client_id`：使用的触发矩阵对应的客户端编号。
- `--output`：指标输出 JSON 文件路径。

若进行零样本分类，将 `--dataset` 改为分类数据集并省略 `--language` 参数即可。

## 8. 查看结果与进一步分析
- 评估日志会在控制台打印触发矩阵嵌入前后的一系列统计信息，可用于检查水印效果。
- 结果 JSON 中包含各项指标，可通过 `clip_benchmark build` 命令汇总到 CSV：
  ```bash
  clip_benchmark build --input_glob "results/*.json" --output summary.csv
  ```
- 若需要重复实验，只需更换模型、数据集或触发矩阵配置并重新运行 `clip_benchmark eval`。

---
完成以上步骤，你就可以在全新环境中成功运行 MFL-Owner，既能评估 CLIP 模型的零样本性能，也能验证自定义的水印触发矩阵。
