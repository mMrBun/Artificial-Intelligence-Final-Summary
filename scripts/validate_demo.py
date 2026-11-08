"""Evaluate the synthetic demo, recording observations rather than assumed gains."""

import json
import statistics
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("KITE-27 的演示和答辩分别有几分钟？", "课程项目手册.pdf", "演示时长为8分钟"),
    ("检索与重排序在评分中占多少分？", "课程项目手册.pdf", "30分"),
    ("GPU 演示机应提前多久预约，最多用多久？", "实验室资源说明.docx", "24小时"),
    ("出现 KB-409 时应如何处理？", "故障排查.md", "重新建立索引"),
    ("上传单个文件超过限制怎么办？", "故障排查.md", "30MB"),
    ("没有配置 DeepSeek 密钥时，本地检索还能用吗？", "故障排查.md", "本地检索实验仍可运行"),
    ("服务重启后资料还能继续检索吗？", "课程项目手册.pdf", "重启后资料仍应可检索"),
    ("表格很大需要切分时，应该保留什么？", "故障排查.md", "重复表头"),
]


def normalize(text):
    return "".join(text.split())


def run():
    report = {
        "note": "小规模虚构演示集，仅作功能验收，不代表通用准确率。",
        "retrieval": [],
        "generation": [],
    }
    with httpx.Client(base_url=BASE, timeout=180) as client:
        warmup = client.post("/api/search", json={"query": "课程项目提交要求", "use_rerank": True})
        warmup.raise_for_status()
        report["warmup_timings"] = warmup.json()["timings"]
        for query, filename, evidence in CASES:
            row = {"query": query, "expected_file": filename, "expected_evidence": evidence}
            for rerank in (False, True):
                response = client.post(
                    "/api/search",
                    json={"query": query, "retrieval_k": 20, "rerank_k": 5, "use_rerank": rerank},
                )
                response.raise_for_status()
                data = response.json()
                matches = [
                    s["filename"] == filename and evidence in normalize(s["text"])
                    for s in data["sources"]
                ]
                row["rerank" if rerank else "dense"] = {
                    "hit_at_1": bool(matches and matches[0]),
                    "hit_at_5": any(matches),
                    "top_file": data["sources"][0]["filename"] if data["sources"] else "",
                    "timings": data["timings"],
                }
            report["retrieval"].append(row)
            print(query, row["dense"]["hit_at_1"], row["rerank"]["hit_at_1"], flush=True)
        for mode in ("dense", "rerank"):
            report[mode + "_summary"] = {
                "hit_at_1": sum(r[mode]["hit_at_1"] for r in report["retrieval"]) / len(CASES),
                "hit_at_5": sum(r[mode]["hit_at_5"] for r in report["retrieval"]) / len(CASES),
                "mean_ms": statistics.mean(
                    r[mode]["timings"]["total_ms"] for r in report["retrieval"]
                ),
            }
        history = []
        for query in [
            "KITE-27 的演示时长是多少？",
            "那答辩呢？",
            "GPU 演示机应提前多久预约，最多用多久？",
            "老师的手机号码是多少？",
        ]:
            response = client.post(
                "/api/chat/stream",
                json={"query": query, "history": history, "retrieval_k": 20, "rerank_k": 5},
            )
            response.raise_for_status()
            events, answer, sources = [], "", []
            for block in response.text.split("\n\n"):
                if not block.strip():
                    continue
                name = next((x[7:] for x in block.splitlines() if x.startswith("event: ")), "")
                payload = next((x[6:] for x in block.splitlines() if x.startswith("data: ")), "{}")
                data = json.loads(payload)
                events.append(name)
                if name == "token":
                    answer += data
                if name == "sources":
                    sources = data["sources"]
                if name == "error":
                    raise RuntimeError(data)
            assert "done" in events and answer
            report["generation"].append(
                {
                    "query": query,
                    "answer": answer,
                    "sources": [
                        {"citation": s["citation"], "filename": s["filename"], "pages": s["pages"]}
                        for s in sources
                    ],
                }
            )
            print(query, answer, flush=True)
            if len(report["generation"]) == 1:
                history = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ]
            else:
                history = []
    path = ROOT / "logs/validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", path, flush=True)


if __name__ == "__main__":
    run()
