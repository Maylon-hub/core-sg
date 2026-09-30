"""Run the entire suite without putting the production source on sys.path."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parents[1]
target = Path(tempfile.mkdtemp(prefix="core-sg-installed-tests-"))
shutil.copy2(source / "pyproject.toml", target / "pyproject.toml")
for folder in ("tests", "benchmarking", "benchmarks", "scripts"):
    for file in (source / folder).rglob("*.py"):
        dest = target / file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
env = dict(os.environ)
env.pop("PYTHONPATH", None)
env["PYTHONDONTWRITEBYTECODE"] = "1"
print("Isolated test inputs:", target, flush=True)
result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", *sys.argv[1:]], cwd=target, env=env)
# Keep failures/inputs inspectable; never remove the user's source or data.
raise SystemExit(result.returncode)
