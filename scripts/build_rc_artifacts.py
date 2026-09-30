"""Build the sdist and exactly one native wheel for the RC host platform."""
from pathlib import Path
import argparse
import os
import platform
import subprocess
import sys
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--reuse-sdist", action="store_true", help="Build from the common, already-created CI sdist")
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
if args.reuse_sdist:
    assert output.is_dir() and len(list(output.iterdir())) == 1, "Use only the common source archive"
else:
    assert not output.exists(), "Use a new artifact directory"
assert sys.version_info[:2] == (3, 11)
assert platform.machine().lower() in ("amd64", "x86_64")
assert sys.platform in ("win32", "linux"), "This RC builds Windows/Linux only"
version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
env = dict(os.environ)
env.pop("PYTHONPATH", None)
if not args.reuse_sdist:
    subprocess.run([sys.executable, "-m", "build", "--sdist", "--outdir", str(output), str(source)], env=env, check=True)
archives = list(output.glob("*.tar.gz"))
assert len(archives) == 1
assert archives[0].name == f"core_sg_mustache-{version}.tar.gz"
subprocess.run([sys.executable, "-m", "cibuildwheel", str(archives[0]), "--output-dir", str(output)], env=env, check=True)
assert len(list(output.glob("*.whl"))) == 1, "Publish cp311 x86-64 only"
print("Built CORE-SG RC", version, "for", sys.platform, platform.machine())
