from pathlib import Path

from docx import Document
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

root = Path("examples/demo")
root.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("DemoChinese", "/mnt/c/Windows/Fonts/simhei.ttf"))
c = canvas.Canvas(str(root / "课程项目手册.pdf"), pagesize=(595, 842))
pages = [
    (
        "智能知识库课程项目手册",
        [
            "以下内容是专门用于演示与测试的虚构课程资料。",
            "一、项目提交要求",
            "项目代号为 KITE-27。演示日期为 2026 年 12 月 18 日。",
            "每组最多 4 人，演示时长为 8 分钟，答辩时长为 4 分钟。",
            "提交材料包括源代码、运行说明、测试报告和 3 分钟演示视频。",
            "二、证据引用规则",
            "回答中的关键结论必须标注资料编号与页码。",
            "如果资料不足，应说明当前知识库没有足够依据，不能编造答案。",
            "多轮问答需要消解代词，但历史回答不能代替原始证据。",
        ],
    ),
    (
        "评分细则与系统边界",
        [
            "三、评分表",
            "评分项目                    分值",
            "文档解析与结构切分          25 分",
            "检索与重排序                30 分",
            "答案依据与来源展示          25 分",
            "界面与可复现性              20 分",
            "四、性能验收",
            "温启动后的检索与重排合计目标为 3 秒以内，不含大模型生成。",
            "第一次查询包含模型加载时间，需单独记录，不能混入温启动数据。",
            "性能目标只是本演示的验收规则，不代表当前程序已经达标。",
        ],
    ),
    (
        "数据保留与演示范围",
        [
            "五、数据管理",
            "删除文档时必须同步清理原文件、解析结果、文本块与对应向量。",
            "内容完全一致的文件应复用已有索引，避免重复入库。",
            "本课程演示没有规定最终论文的字数，也没有提供教师手机号码。",
            "六、检索策略",
            "初始候选默认取 20 块，经过重排序后选取 5 块作为回答依据。",
            "相似度与重排分数用于排序，不是答案正确概率。",
            "课程知识库使用 Chroma 本地持久化，重启后资料仍应可检索。",
        ],
    ),
]
for title, lines in pages:
    c.setFont("DemoChinese", 20)
    c.drawString(50, 785, title)
    y = 737
    for line in lines:
        c.setFont("DemoChinese", 12)
        c.drawString(50, y, line)
        y -= 36
    c.showPage()
c.save()
doc = Document()
doc.add_heading("实验室资源说明", 0)
doc.add_paragraph("以下为虚构演示资料，与真实实验室政策无关。")
doc.add_heading("设备与模型", 1)
doc.add_paragraph(
    "演示工作站使用 RTX 4060 Laptop，显存 8GB。嵌入模型与重排序模型使用 Qwen3 0.6B，本地 PDF 解析使用 MinerU Basic。"
)
doc.add_heading("设备预约", 1)
table = doc.add_table(rows=1, cols=3)
for cell, text in zip(table.rows[0].cells, ["资源", "提前预约", "最长使用时间"]):
    cell.text = text
for row in [
    ["GPU 演示机", "24 小时", "2 小时"],
    ["录屏工位", "12 小时", "1 小时"],
    ["讨论室", "6 小时", "3 小时"],
]:
    for cell, text in zip(table.add_row().cells, row):
        cell.text = text
doc.add_heading("旧项目对照", 1)
doc.add_paragraph(
    "去年的项目代号为 KITE-17，演示时长 5 分钟，答辩 2 分钟。该规定只适用于旧项目，不能用于今年的 KITE-27。"
)
doc.save(root / "实验室资源说明.docx")
(root / "故障排查.md").write_text(
    """# 智能知识库故障排查
以下为虚构演示资料。

## 错误 KB-413：文件过大
单个上传文件的默认上限为 30 MB。遇到 KB-413，应压缩或拆分文件后重新上传。

## 错误 KB-401：生成接口认证失败
检查 .env 中的 DEEPSEEK_API_KEY 是否正确。密钥缺失只影响生成；本地检索实验仍可运行。不要把密钥写进源代码或日志。

## 错误 KB-409：索引配置不一致
更换 embedding 模型后应重新建立索引。旧模型生成的向量不能与新模型的查询向量混用，即便维度相同。

## 扫描文件与表格
扫描 PDF 需要 OCR。表格应保持行列关系，大表切分后应重复表头。Markdown 和 TXT 没有稳定页码，应展示文件与章节。

## 候选数量与重排序
召回不足时可以增大初始候选数量，再由重排序筛选。不要把候选全部塞进生成模型；无关段落会增加延迟和干扰。
""",
    encoding="utf-8",
)
print("Created 3 clearly marked synthetic demo documents.")
