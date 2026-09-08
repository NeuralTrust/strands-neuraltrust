"""Run lint, formatting, type checks, and the offline test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "examples", "scripts"],
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "examples", "scripts"],
        [sys.executable, "-m", "mypy"],
        [sys.executable, "-m", "pytest", "-W", "error"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
