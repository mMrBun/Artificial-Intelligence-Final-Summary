"""Run the local MinerU SDK in a child process so GPU memory is reclaimed."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .config import ROOT, Settings
from .schemas import Element


class MinerUParser:
    def __init__(self, settings: Settings):
        self.settings = settings

    def parse(self, path: Path, output: Path) -> list[Element]:
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "parser.log"
        with log_path.open("w") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "ragkb.parser_worker", str(path), str(output)],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
        if completed.returncode:
            tail = log_path.read_text(errors="replace")[-1500:]
            raise RuntimeError(f"本地文档解析失败，详见 {log_path.name}。\n{tail}")
        data = json.loads((output / "elements.json").read_text())
        elements = [Element(**item) for item in data]
        if not any(e.text.strip() for e in elements):
            raise ValueError("文档没有可索引内容，请检查文件或 OCR 结果。")
        return elements

    @staticmethod
    def save_elements(elements, output: Path):
        (output / "elements.json").write_text(
            json.dumps([asdict(e) for e in elements], ensure_ascii=False), encoding="utf-8"
        )
