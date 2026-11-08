# 知页：RAG 课程知识库

FastAPI + Gradio + MinerU 本地 SDK + Qwen3 + Chroma + DeepSeek。
所有代码在服务器 /home/bun/projects/rag-knowledge-base，Python 3.12，uv 管理依赖。

## 访问与运行

```bash
cd /home/bun/projects/rag-knowledge-base
bash scripts/service.sh start
bash scripts/service.sh status
bash scripts/service.sh stop
bash scripts/service.sh restart
```

前台运行：bash scripts/start.sh。只启动一个实例和一个 worker，API/UI 共用模型和 Chroma。

WSL 内访问：
- 界面：http://127.0.0.1:8000/ui/
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health

其他电脑建议通过 SSH 通道访问，不依赖 Windows 端口转发：

```bash
ssh -N -L 18000:127.0.0.1:8000 bun@192.168.31.36
```

保持终端运行，再打开 http://127.0.0.1:18000/ui/。
本次浏览器验证使用此方式，直接访问局域网 8000 端口未成功。

## 配置与依赖

首次部署复制 .env.example 为 .env，填写：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

deepseek-flash 是官方 API 的 V4.1 Flash 调用名。
密钥和 DeepSeek 配置每次调用重新读取，修改后无需重启；其余配置修改后重启。

重建环境与显式下载已确认模型：

```bash
uv sync --locked
.venv/bin/python scripts/download_models.py
```

模型优先从 ModelScope 下载，所有权重及下载缓存都在 models/：

```text
models/
├── Qwen3-Embedding-0.6B/
├── Qwen3-Reranker-0.6B/
├── MinerU-Basic/
├── _cache/
└── manifest.json
```

模型目录总计约 3.3GB。日常运行仅加载本地权重，不隐式下载。
MinerU 配置位于 config/mineru.yaml；迁移路径时修改 model.base_dir。
DeepSeek 使用 API，没有本地权重。依赖锁定在 uv.lock，PyTorch/CUDA 另需磁盘空间。

## 处理流程

1. 上传 PDF（含扫描件）、DOCX、Markdown、UTF-8 TXT、PNG/JPG。
2. PDF/图片通过 MinerU Basic 本地解析；DOCX 使用 Flash；MD/TXT 直接读取。
3. 根据标题、段落和表格切分，默认 512 tokens，标题和文件名前缀也计入预算。
4. 超长元素分割时保留最多 64 tokens 重叠；普通结构块不机械重叠。大表按行切分并重复表头。
5. Qwen3-Embedding-0.6B 向量化，Chroma 持久化；默认召回 20 块。
6. Qwen3-Reranker-0.6B 重排并保留 5 块，DeepSeek 流式回答并标注来源。

每块保存文件名、章节、序号、PDF 原始页码。DOCX/MD/TXT 不编造固定页码。
图片仅支持 OCR 和图注文本，不做图像语义向量检索。
文档在本地处理；问题、必要历史及命中的文本片段发送到 DeepSeek 生成答案。

## 面向对象模块（ragkb/）

| 文件 | 类与职责 |
|---|---|
| config.py / schemas.py | Settings、Element、Chunk 和接口参数 |
| parsing.py / parser_worker.py | MinerUParser、本地 SDK 子进程与结果适配 |
| chunking.py | StructureChunker，结构与 token 预算 |
| model_manager.py | ModelManager，GPU 工作互斥与资源释放 |
| embedding.py / reranker.py | EmbeddingService、Reranker |
| vector_store.py | ChromaStore，向量持久化与检索 |
| repository.py | DocumentRepository，SQLite 状态和分块预览 |
| ingestion.py | IngestionService，上传、去重、入库和删除 |
| retrieval.py | Retriever，召回、重排及耗时记录 |
| llm.py / qa.py | DeepSeekClient、QAService |
| container.py | AppContainer，共享唯一服务实例 |
| api.py | FastAPI 路由、SSE |
| ui.py / ui_actions.py | DemoUI、UIActions |
| main.py | 应用生命周期与启动 |

导入文档时先释放检索模型，再用子进程运行 MinerU，解析后退出以回收显存。
GPU 工作串行处理，导入期间查询可能等待，适配 8GB 显存。

Chroma 保存向量与来源，SQLite 保存文档状态与预览。
完整入库后才标记 ready；失败清理部分向量；重启后未完成任务标记失败，可重新上传。
内容完全相同的文件按 SHA-256 去重。更换 embedding 模型后需重新建索引。
删除会清理原文件、解析结果、预览数据与向量。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | /api/health | 状态、模型就绪、密钥是否配置 |
| POST | /api/documents | multipart 上传，返回 ID |
| GET | /api/documents | 文档列表 |
| GET | /api/documents/{id} | 处理进度 |
| GET | /api/documents/{id}/chunks | 分块，支持 offset/limit |
| GET | /api/documents/{id}/file | 原文件 |
| DELETE | /api/documents/{id} | 删除文档及数据 |
| POST | /api/search | 检索和可选重排序 |
| POST | /api/chat/stream | 流式问答 |

SSE 事件：sources、多个 token、done；中途失败发送 error，不发送正常完成。
密钥缺失返回 503；本地检索无需密钥。文档列表不暴露服务器文件路径。
日志：logs/server.log；解析日志：data/parsed/<文档ID>/parser.log。
密钥和 models/data/logs 均由 .gitignore 排除。

## 验证与演示

```bash
.venv/bin/pytest -q
.venv/bin/ruff check ragkb scripts tests tests_support.py
.venv/bin/python scripts/validate_demo.py
.venv/bin/python scripts/validate_lifecycle.py
```

examples/demo/ 内有 3 份明确标注为虚构的课程资料。
create_demo_documents.py 可重新生成，PDF 嵌入本服务器的 Windows 中文字体。
validate_demo.py 对 8 道问题比较是否重排，并实际调用 DeepSeek 验证多轮、表格和拒答。
validate_lifecycle.py 验证去重、删除、扫描件 OCR，清理自己建立的临时文档。
结果保存在 logs/validation.json、logs/lifecycle_validation.json。
具体实测结果与演示步骤见 VALIDATION.md。

相似度与重排分数用于排序，不是答案正确概率。
小型演示集只能用于功能验收，不能推断通用准确率，也不能保证重排始终提升效果。

## 上游资料

- Qwen：https://github.com/QwenLM/Qwen3-Embedding
- MinerU SDK：https://opendatalab.github.io/MinerU/usage/sdk_api/
- MinerU 模型：https://opendatalab.github.io/MinerU/usage/model_source/
- DeepSeek API：https://api-docs.deepseek.com/

