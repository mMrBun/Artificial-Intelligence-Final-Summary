# 本地模型目录

- Qwen3-Embedding-0.6B/：向量模型和 tokenizer
- Qwen3-Reranker-0.6B/：重排序模型和 tokenizer
- MinerU-Basic/：MinerU Basic 的 Torch 小模型包
- _cache/：下载器及推理库缓存，同样限制在此根目录内
- manifest.json：显式下载完成后的来源记录

运行 scripts/download_models.py 从 ModelScope 准备已确认模型。
日常运行开启本地加载，不会静默下载其他模型。
DeepSeek 使用 API，因此此处没有 DeepSeek 权重。

