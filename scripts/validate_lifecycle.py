import io
import json
import time
from pathlib import Path

import httpx
import pypdfium2 as pdfium

root = Path(__file__).resolve().parents[1]
report = {}
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=180) as client:

    def upload(name, data):
        r = client.post("/api/documents", files={"file": (name, data)})
        r.raise_for_status()
        return r.json()

    def wait(doc_id):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            d = client.get("/api/documents/" + doc_id).json()
            if d["status"] in {"ready", "failed"}:
                assert d["status"] == "ready", d
                return d
            time.sleep(1)
        raise TimeoutError("Document ingestion timed out")

    first = upload(
        "临时生命周期测试.txt", "生命周期测试专用编号 ORBIT-953。验证完毕自动清理。".encode()
    )
    wait(first["id"])
    duplicate = upload(
        "重命名生命周期测试.txt", "生命周期测试专用编号 ORBIT-953。验证完毕自动清理。".encode()
    )
    assert duplicate["duplicate"] and duplicate["id"] == first["id"]
    report["duplicate_upload"] = "passed"
    r = client.delete("/api/documents/" + first["id"])
    r.raise_for_status()
    assert client.get("/api/documents/" + first["id"]).status_code == 404
    results = client.post("/api/search", json={"query": "ORBIT-953", "use_rerank": False}).json()
    assert all(c["document_id"] != first["id"] for c in results["candidates"])
    assert not list((root / "data/uploads").glob(first["id"] + "*"))
    report["delete_removes_vectors_and_file"] = "passed"

    pdf = pdfium.PdfDocument(str(root / "examples/demo/课程项目手册.pdf"))
    page = pdf[0]
    bitmap = page.render(scale=2)
    image = bitmap.to_pil().convert("RGB")
    payload = io.BytesIO()
    image.save(payload, format="PDF", resolution=144)
    bitmap.close()
    page.close()
    pdf.close()
    scanned = upload("临时扫描PDF测试.pdf", payload.getvalue())
    doc = wait(scanned["id"])
    chunks = client.get("/api/documents/" + scanned["id"] + "/chunks").json()
    combined = "".join(c["text"] for c in chunks)
    assert "KITE" in combined and "8" in combined, combined
    assert all(c["pages"] == [1] for c in chunks)
    report["scanned_pdf_ocr"] = {"status": "passed", "chunks": len(chunks)}
    client.delete("/api/documents/" + scanned["id"]).raise_for_status()
(root / "logs/lifecycle_validation.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)
print(json.dumps(report, ensure_ascii=False), flush=True)
