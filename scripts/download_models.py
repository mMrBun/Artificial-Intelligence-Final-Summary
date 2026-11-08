"""Explicit, resumable model preparation. Runtime never downloads models."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkb.config import ROOT, Settings


def main():
    settings = Settings()
    settings.prepare()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)
    os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
    from modelscope.hub.snapshot_download import snapshot_download

    models = {
        "Qwen/Qwen3-Embedding-0.6B": settings.embedding_path,
        "Qwen/Qwen3-Reranker-0.6B": settings.reranker_path,
    }
    for repo, destination in models.items():
        print(f"Downloading {repo} -> {destination}", flush=True)
        snapshot_download(
            repo, local_dir=str(destination), ignore_patterns=["*.bin", "*.gguf", "*.onnx"]
        )
        print(f"Ready: {repo}", flush=True)
    subprocess.run(
        [
            str(ROOT / ".venv/bin/mineru-kit"),
            "models",
            "download",
            "--tier",
            "basic",
            "--small-backend",
            "torch",
            "--source",
            "modelscope",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(ROOT / ".venv/bin/mineru-kit"),
            "models",
            "verify",
            "--tier",
            "basic",
            "--small-backend",
            "torch",
        ],
        check=True,
    )
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": "modelscope",
        "embedding": "Qwen/Qwen3-Embedding-0.6B",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "mineru": "MinerU-4_models_torch",
        "mineru_tier": "basic",
        "model_root": str(settings.model_root),
    }
    (settings.model_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("All requested local models are ready.", flush=True)


if __name__ == "__main__":
    main()
