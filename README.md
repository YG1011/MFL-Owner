# MFL-Owner

### Install Benchmarks
```bash
pip install benchmarks/CLIP_benchmark
```

### Sample running code for zero-shot evaluation with MFL-Owner:
```bash
# zero-shot retrieval 
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
                    
# zero-shot classification 
clip_benchmark eval --dataset=imagenet1k \
                    --pretrained=openai \
                    --model=ViT-L-14 \
                    --output=result.json \
                    --batch_size=64 \
                    --trigger_num=512 \
                    --watermark_dim=768 \
                    --watermark_dir "/root/watermark"  \
                    --dataset_root "/root" \
                    --client_id= 0
```