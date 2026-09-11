import shutil
import subprocess
from pathlib import Path

from src.microgrid_model import run


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    run(project_root)
    node = shutil.which("node")
    if node is None:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
        node = str(bundled) if bundled.exists() else None
    if node is None:
        raise RuntimeError("未找到Node.js，无法将求解载荷写入Excel结果模板")
    subprocess.run(
        [node, str(project_root / "scripts/build_result_workbooks.mjs")],
        cwd=project_root,
        check=True,
    )
