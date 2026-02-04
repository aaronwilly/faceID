"""
Download all project dependencies as wheels into the wheels/ folder.
Run this once while ONLINE. Then you can install offline with install_offline.py (or install_offline.ps1).

Usage (from project root):
  python -m venv venv
  venv\Scripts\activate
  pip install pip --upgrade
  python scripts/download_wheels.py
"""
from __future__ import annotations

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEELS_DIR = os.path.join(PROJECT_ROOT, "wheels")
REQUIREMENTS_OFFLINE = os.path.join(PROJECT_ROOT, "requirements-offline.txt")
IP_ADAPTER_GIT = "git+https://github.com/tencent-ailab/IP-Adapter.git"


def run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> int:
    env = env or os.environ.copy()
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd or PROJECT_ROOT, env=env)


def main() -> int:
    os.makedirs(WHEELS_DIR, exist_ok=True)
    if not os.path.isfile(REQUIREMENTS_OFFLINE):
        print(f"Missing {REQUIREMENTS_OFFLINE}")
        return 1

    print("Step 0: Downloading build deps (setuptools, wheel) so sdist builds work offline ...")
    if run([
        sys.executable, "-m", "pip", "download",
        "-d", WHEELS_DIR,
        "setuptools", "wheel",
    ]) != 0:
        return 1

    print("Step 1: Downloading PyPI packages as wheels into wheels/ ...")
    # Download for current platform (default). For torch CUDA, use extra index if needed.
    if run([
        sys.executable, "-m", "pip", "download",
        "-d", WHEELS_DIR,
        "-r", REQUIREMENTS_OFFLINE,
    ]) != 0:
        print("Tip: If torch/torchvision fail, try with CUDA index:")
        print('  pip download -d wheels -r requirements-offline.txt --extra-index-url https://download.pytorch.org/whl/cu126')
        return 1

    print("Step 2: Installing from wheels so we have deps to build IP-Adapter ...")
    if run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", WHEELS_DIR,
        "-r", REQUIREMENTS_OFFLINE,
    ]) != 0:
        return 1

    print("Step 3: Building IP-Adapter wheel from git into wheels/ ...")
    if run([
        sys.executable, "-m", "pip", "wheel",
        IP_ADAPTER_GIT,
        "-w", WHEELS_DIR,
        "--no-deps",
    ]) != 0:
        print("IP-Adapter wheel build failed. You can still install other deps offline.")
        return 1

    print("Done. Wheels are in wheels/. For offline install run: python scripts/install_offline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
