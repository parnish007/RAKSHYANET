#!/usr/bin/env python3
"""
RakshyaNet Automated Test Runner
Verifies JSON files and runs the full pytest suite.
Usage: python scripts/run_tests.py
"""
import subprocess
import sys
import os
from pathlib import Path

# Colour codes (disabled on Windows if no ANSI support)
GREEN  = "\033[92m" if sys.platform != "win32" or os.environ.get("TERM") else ""
RED    = "\033[91m" if sys.platform != "win32" or os.environ.get("TERM") else ""
YELLOW = "\033[93m" if sys.platform != "win32" or os.environ.get("TERM") else ""
RESET  = "\033[0m"  if sys.platform != "win32" or os.environ.get("TERM") else ""

PROJECT_ROOT = Path(__file__).parent.parent

JSON_FILES = [
    "backend/data/nepal_villages.json",
    "backend/data/fleet_config.json",
    "backend/data/terrain_graph.json",
    "backend/data/config.json",
    "demo/mock_news_timeline.json",
]


def run(cmd: str, description: str, capture: bool = False) -> subprocess.CompletedProcess:
    print(f"\n{'='*62}")
    print(f"  {description}")
    print(f"{'='*62}")
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True, cwd=PROJECT_ROOT
    )
    if not capture:
        pass  # output already streamed to terminal
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"{YELLOW}STDERR:{RESET}", result.stderr)

    if result.returncode != 0:
        print(f"\n{RED}FAILED:{RESET} {description}")
        sys.exit(1)
    print(f"{GREEN}OK:{RESET} {description}")
    return result


def pip() -> str:
    """Return path to pip inside venv (or system pip)."""
    venv = PROJECT_ROOT / "venv"
    if sys.platform == "win32":
        candidate = venv / "Scripts" / "pip.exe"
    else:
        candidate = venv / "bin" / "pip"
    return str(candidate) if candidate.exists() else "pip"


def pytest_cmd() -> str:
    venv = PROJECT_ROOT / "venv"
    if sys.platform == "win32":
        candidate = venv / "Scripts" / "pytest.exe"
    else:
        candidate = venv / "bin" / "pytest"
    return str(candidate) if candidate.exists() else "pytest"


def main():
    os.chdir(PROJECT_ROOT)

    print(f"\n{'='*62}")
    print(f"  RAKSHYANET AUTOMATED TEST SUITE")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'='*62}\n")

    # 1. Setup venv if needed
    venv_path = PROJECT_ROOT / "venv"
    if not venv_path.exists():
        run(f"{sys.executable} -m venv venv", "Creating virtual environment")
    else:
        print(f"{GREEN}OK:{RESET} Virtual environment already exists")

    # 2. Install dependencies
    run(f"{pip()} install -q --upgrade pip", "Upgrading pip", capture=True)
    run(f"{pip()} install -q -r requirements.txt", "Installing dependencies", capture=True)

    # 3. JSON validity
    print(f"\n{'='*62}")
    print(f"  JSON VALIDITY CHECKS")
    print(f"{'='*62}")
    for path in JSON_FILES:
        full_path = PROJECT_ROOT / path
        if not full_path.exists():
            print(f"{YELLOW}SKIP:{RESET} {path} (not found)")
            continue
        result = subprocess.run(
            [sys.executable, "-c", f"import json; json.load(open(r'{full_path}'))"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"{GREEN}OK:{RESET}   {path}")
        else:
            print(f"{RED}FAIL:{RESET}  {path}")
            print(result.stderr)
            sys.exit(1)

    # 4. Pytest
    run(
        f"{pytest_cmd()} backend/tests/ -v --tb=short",
        "Running full test suite"
    )

    print(f"\n{'='*62}")
    print(f"  ALL CHECKS PASSED")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
