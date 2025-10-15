# 项目运行流程概览

本文档总结了 MFL-Owner 项目从准备触发矩阵到执行带水印验证的零样本评估的完整流程。

## 1. 环境准备
1. 克隆仓库并进入根目录。
2. 安装扩展后的 `clip_benchmark` 包：
   ```bash
   pip install benchmarks/CLIP_benchmark
   ```
3. 准备好下游评估所需的数据集（如 `flickr30k`、`imagenet1k` 等），并确保命令中的 `--dataset_root` 指向正确位置。

## 2. 触发矩阵训练
`benchmarks/trigger/training.py` 用于针对多客户端嵌入训练水印触发矩阵 `W_align`：
1. 入口处读取命令行参数，确定触发器数量和训练方法等配置。脚本默认在 GPU 上运行（若可用）。【F:benchmarks/trigger/training.py†L24-L31】
2. 通过 `get_emb` 载入每个客户端的图像嵌入 `Y` 与文本嵌入 `T`，形状均为 `[trigger_num, 768]`。【F:benchmarks/trigger/training.py†L33-L36】
3. 对每个客户端：
   - 若磁盘已存在对应的触发矩阵则直接加载；否则随机初始化一个 768×768 的正交矩阵并以 Adam 优化器训练 1000 轮，最小化 `||W_align @ Y - W_align @ T||_F`，同时可选地施加近似正交约束。【F:benchmarks/trigger/training.py†L38-L79】
   - 训练完成后将矩阵保存到 `/root/trigger/results/w_matrix/noisy_w/` 下并输出初步的相似度与距离验证指标。【F:benchmarks/trigger/training.py†L81-L120】
4. 验证环节会比较水印前后以及“逆变换”恢复后的特征余弦相似度与欧氏距离，用于确认触发矩阵既能嵌入水印也能还原原始嵌入。【F:benchmarks/trigger/training.py†L92-L119】

> **产物**：`trigger_mat_c{i}_{trigger_num}_{method}.pth`。评估阶段会按矩阵维度与触发器数量组合读取。

## 3. 执行带水印的评估
命令行入口位于 `clip_benchmark/cli.py`，核心流程如下：
1. `clip_benchmark eval ...` 会解析评估相关参数（模型、数据集、任务、批大小等）以及水印参数 `--watermark_dir`、`--watermark_dim`、`--trigger_num`。【F:benchmarks/CLIP_benchmark/clip_benchmark/cli.py†L18-L86】
2. `main_eval` 会枚举所有模型×数据集×语言组合，逐一调用 `run(args)`。【F:benchmarks/CLIP_benchmark/clip_benchmark/cli.py†L106-L166】
3. `run` 函数根据任务类型加载模型、构建数据集与 DataLoader，并在评估阶段注入触发矩阵：
   - 零样本分类：在构建类原型和图像特征时右乘触发矩阵，并在每个 batch 内、以及总体上打印水印嵌入与逆变换的相似度验证信息。【F:benchmarks/CLIP_benchmark/clip_benchmark/metrics/zeroshot_classification.py†L17-L120】
   - 零样本检索：对图像与文本特征同样右乘触发矩阵并归一化，记录水印嵌入及逆向恢复的验证指标，随后计算 Recall@K 等检索指标。【F:benchmarks/CLIP_benchmark/clip_benchmark/metrics/zeroshot_retrieval.py†L41-L150】
4. 每次评估结束后，`run` 会将指标写入 `--output` 指定的 JSON 文件（默认 `result.json`）。若需要可使用 `clip_benchmark build` 将多个结果汇总为 CSV。【F:benchmarks/CLIP_benchmark/clip_benchmark/cli.py†L168-L244】

## 4. 典型命令示例
以 README 中的零样本检索为例：
```bash
clip_benchmark eval --model ViT-L-14 \
                    --pretrained laion2b_s32b_b82k \
                    --dataset=flickr30k \
                    --output=result.json \
                    --batch_size=64  \
                    --language=en \
                    --trigger_num=512 \
                    --watermark_dim=768 \
                    --client_id=0 \
                    --dataset_root "/root" \
                    --watermark_dir "/root/watermark"
```
命令会在加载模型与数据集后读取 `/root/watermark/768/trigger_mat_512.pth`，将其应用于图像和文本特征并输出检索性能以及水印验证日志。

## 5. 总结
整个流程可概括为：
1. 训练或准备好触发矩阵；
2. 使用扩展后的 `clip_benchmark eval` 执行任务评估；
3. 在评估过程中水印矩阵会被自动载入、嵌入与验证；
4. 评估结果与验证日志可用于分析模型性能及水印效果。
