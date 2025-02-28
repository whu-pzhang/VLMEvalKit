# 遥感多模态大模型性能测试

运行环境：

|         |             |
| ------- | ----------- |
| GPU     | 2*RTX8000   |
| Pytorch | 2.1.2+cu121 |
| CUDA    | 12.1        |
| VLMEval | 0.1.0       |


## Scene Classification

command:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data2/hf_models/ \
torchrun --nproc-per-node=2 run.py --data AID --model Qwen2.5-VL-3B-Instruct
```


| Model                      | Acc   | TF version |
| -------------------------- | ----- | ---------- |
| InternVL2.5-1B             | 61.50 | 4.37.2     |
| InternVL2.5-1B-MPO         | 59.88 | 4.37.2     |
| Mini-InternVL-Chat-4B-V1-5 | 56.86 | 4.37.2     |
| Qwen2.5-VL-3B-Instruct     |       | 4.49.0     |



Note:

Qwen2.5-VL 正常运行步骤：
1. 需安装 `qwen-vl-utils`: `pip install qwen-vl-utils[decord]`
2. 需在 `run.py` 文件加上以下两行：

```python
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
```

以免出现如下报错：

```bash
RuntimeError: cutlassF: no kernel found to launch!
```
