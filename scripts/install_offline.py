"""
Install all dependencies from the wheels/ folder only (no network).
Use this on a machine with no internet after copying the project + wheels/ folder.

Usage (from project root, with venv activated):
  python scripts/install_offline.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEELS_DIR = os.path.join(PROJECT_ROOT, "wheels")
REQUIREMENTS_OFFLINE = os.path.join(PROJECT_ROOT, "requirements-offline.txt")


def main() -> int:
    if not os.path.isdir(WHEELS_DIR):
        print(f"Missing wheels folder: {WHEELS_DIR}")
        print("Run download_wheels.py first (while online) to populate it.")
        return 1

    # So build isolation (for sdists like insightface) also uses wheels only
    env = os.environ.copy()
    env["PIP_NO_INDEX"] = "1"
    env["PIP_FIND_LINKS"] = WHEELS_DIR

    # Install from requirements-offline.txt using only wheels
    if os.path.isfile(REQUIREMENTS_OFFLINE):
        r = subprocess.call([
            sys.executable, "-m", "pip", "install",
            "--no-index", "--find-links", WHEELS_DIR,
            "-r", REQUIREMENTS_OFFLINE,
        ], cwd=PROJECT_ROOT, env=env)
        if r != 0:
            return r

    # Install ip_adapter wheel (built by download_wheels.py)
    ip_whl = os.path.join(WHEELS_DIR, "ip_adapter-*.whl")
    if glob.glob(ip_whl):
        r = subprocess.call([
            sys.executable, "-m", "pip", "install",
            "--no-index", "--find-links", WHEELS_DIR,
            "ip_adapter",
        ], cwd=PROJECT_ROOT, env=env)
        if r != 0:
            return r
    else:
        print("No ip_adapter wheel found in wheels/. Install it from wheels manually if needed.")

    print("Offline install finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
