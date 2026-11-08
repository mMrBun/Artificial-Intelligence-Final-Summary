import gradio as gr

from .ui_actions import UIActions


class DemoUI:
    def __init__(self, services):
        self.actions = UIActions(services)

    def build(self):
        a = self.actions
        with gr.Blocks(title="知页 · RAG 知识库") as demo:
            gr.Markdown("# 知页 · 课程知识库\n上传资料，追溯证据，观察检索如何改善回答。")
            status = gr.Markdown(a.health())
            with gr.Tab("知识问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=440, label="基于资料的回答", sanitize_html=True)
                        question = gr.Textbox(
                            label="你的问题",
                            placeholder="先在「文档管理」上传资料，再提出问题。",
                            lines=2,
                        )
                        with gr.Row():
                            send = gr.Button("发送问题", variant="primary")
                            clear = gr.Button("新对话")
                    with gr.Column(scale=1):
                        gr.Markdown("### 检索设置\n先扩大候选范围，再筛选最相关的证据。")
                        recall_k = gr.Slider(5, 50, value=20, step=1, label="初始召回数量")
                        final_k = gr.Slider(1, 10, value=5, step=1, label="最终证据数量")
                        rerank = gr.Checkbox(value=True, label="使用重排序")
                        gr.Markdown("回答中的 [1]、[2] 对应下方来源。资料不足时应明确说明。")
                history = gr.State([])
                sources = gr.Dataframe(
                    headers=["来源", "文件", "页码", "章节", "相似度", "重排分数", "原文"],
                    interactive=False,
                    wrap=True,
                    label="回答依据",
                )
                links = gr.Markdown()
                with gr.Accordion("检索过程与耗时", open=False):
                    trace = gr.JSON(label="本轮检索记录")
                outputs = [chatbot, history, question, sources, trace, links]
                for event in (send.click, question.submit):
                    event(
                        a.chat,
                        [question, history, recall_k, final_k, rerank],
                        outputs,
                        concurrency_limit=1,
                        concurrency_id="chat",
                    )
                clear.click(lambda: ([], [], "", [], {}, ""), outputs=outputs, queue=False)
                chatbot.clear(lambda: ([], [], "", [], {}, ""), outputs=outputs, queue=False)

            with gr.Tab("文档管理"):
                gr.Markdown(
                    "支持 PDF（含扫描件）、DOCX、Markdown、UTF-8 TXT、PNG/JPG。"
                    "图片仅做 OCR 与图注检索；不提供图像语义检索。"
                )
                upload_files = gr.File(
                    label="选择文档",
                    file_count="multiple",
                    type="filepath",
                    file_types=[".pdf", ".docx", ".md", ".txt", ".png", ".jpg", ".jpeg"],
                )
                with gr.Row():
                    upload = gr.Button("上传并建立索引", variant="primary")
                    refresh = gr.Button("刷新文档列表")
                upload_status = gr.Textbox(label="处理进度", interactive=False, lines=3)
                documents = gr.Dataframe(
                    headers=["文件", "状态", "当前阶段", "块数", "错误"],
                    interactive=False,
                    wrap=True,
                    label="知识库文档",
                )
                selection = gr.Dropdown(label="选择要查看或删除的文档", choices=[])
                with gr.Row():
                    preview = gr.Button("查看分块")
                    delete = gr.Button("删除所选文档及索引", variant="stop")
                preview_status = gr.Markdown()
                chunk_table = gr.Dataframe(
                    headers=["序号", "类型", "章节", "页码", "Tokens", "内容"],
                    interactive=False,
                    wrap=True,
                    label="结构化文本块",
                )
                upload.click(
                    a.upload,
                    [upload_files],
                    [upload_status, documents, selection, status],
                    concurrency_limit=1,
                    concurrency_id="upload",
                )
                refresh.click(a.documents, outputs=[documents, selection], queue=False)
                refresh.click(a.health, outputs=status, queue=False)
                preview.click(a.preview, [selection], [chunk_table, preview_status], queue=False)
                delete.click(
                    a.delete,
                    [selection],
                    [upload_status, documents, selection, chunk_table, status],
                )
                demo.load(a.documents, outputs=[documents, selection])
                demo.load(a.health, outputs=status)

            with gr.Tab("检索实验"):
                gr.Markdown(
                    "无需 DeepSeek 密钥。查看初始相似度排序与重排序后的变化，"
                    "使用同一问题切换开关进行比较。分数用于排序，不代表答案正确概率。"
                )
                query = gr.Textbox(label="检索问题")
                with gr.Row():
                    experiment_recall = gr.Slider(5, 50, value=20, step=1, label="初始召回")
                    experiment_final = gr.Slider(1, 10, value=5, step=1, label="最终保留")
                    experiment_rerank = gr.Checkbox(value=True, label="启用重排序")
                search = gr.Button("运行检索", variant="primary")
                results = gr.Dataframe(
                    headers=["召回排名", "最终排名", "文件", "页码", "相似度", "重排分数", "原文"],
                    interactive=False,
                    wrap=True,
                    label="候选片段对比",
                )
                details = gr.JSON(label="检索耗时与完整结果")
                search.click(
                    a.search,
                    [query, experiment_recall, experiment_final, experiment_rerank],
                    [results, details],
                    concurrency_limit=1,
                )
            gr.Markdown("FastAPI · Gradio · MinerU · Qwen3 · Chroma · DeepSeek")
        return demo.queue(default_concurrency_limit=2, max_size=20)
