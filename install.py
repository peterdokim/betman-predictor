"""Install dependencies and generate simple launchers for the Betman predictor."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


RECOMMENDED_PYTHON = (3, 11)


def is_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or hasattr(sys, "real_prefix")


def guess_venv_python(venv_path: str) -> Optional[str]:
    root = Path(venv_path)
    if platform.system() == "Windows":
        candidate = root / "Scripts" / "python.exe"
        return str(candidate) if candidate.exists() else None

    for relative in ("bin/python3", "bin/python"):
        candidate = root / relative
        if candidate.exists():
            return str(candidate)
    return None


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("-->", " ".join(command))
    return subprocess.run(command, check=check)


def read_requirements(requirements_path: Path) -> list[str]:
    if not requirements_path.exists():
        return []
    with requirements_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def create_run_bat(project_dir: Path, venv_path: Optional[str]) -> None:
    lines = [
        "@echo off",
        'cd /d "%~dp0"',
        "",
    ]
    if venv_path:
        lines.extend(
            [
                "REM Auto-activate the selected venv",
                f'call "{Path(venv_path) / "Scripts" / "activate.bat"}"',
                "",
            ]
        )
    lines.extend(
        [
            "cmd /k \"python app.py\"",
            "",
        ]
    )
    target = project_dir / "run_app.bat"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created: {target}")


def create_run_sh(project_dir: Path, venv_path: Optional[str]) -> None:
    lines = [
        "#!/bin/bash",
        'cd "$(dirname "$0")"',
        "",
    ]
    if venv_path:
        lines.extend(
            [
                f'source "{Path(venv_path) / "bin" / "activate"}"',
                "",
            ]
        )
    lines.extend(["python3 app.py", ""])
    target = project_dir / "run_app.sh"
    target.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(target, 0o755)
    except OSError:
        pass
    print(f"Created: {target}")


def pause_if_double_clicked() -> None:
    try:
        input("\nDone. Press ENTER to close this window...")
    except EOFError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Betman predictor requirements.")
    parser.add_argument(
        "--venv",
        type=str,
        default=None,
        help="path to an existing virtual environment to install into",
    )
    args = parser.parse_args()

    interpreter = sys.executable
    python_version = sys.version_info[:3]
    print("-" * 68)
    print("Betman Predictor Installer")
    print("-" * 68)
    print(f"Interpreter : {interpreter}")
    print(f"Python      : {python_version[0]}.{python_version[1]}.{python_version[2]}")
    print(f"Environment : {'Virtualenv' if is_venv() else 'System'}")
    if args.venv:
        print(f"--venv      : {args.venv}")
    print("-" * 68)

    if python_version[:2] != RECOMMENDED_PYTHON:
        print(
            f"Warning: Python {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]} "
            f"is recommended. Continuing anyway.\n"
        )

    if args.venv:
        chosen_python = guess_venv_python(args.venv)
        if not chosen_python:
            print("Could not locate a Python executable inside the provided venv.")
            pause_if_double_clicked()
            raise SystemExit(1)
    else:
        chosen_python = interpreter

    project_dir = Path(__file__).resolve().parent
    requirements = read_requirements(project_dir / "requirements.txt")
    pip_command = [chosen_python, "-m", "pip"]

    try:
        run(pip_command + ["install", "--upgrade", "pip", "setuptools", "wheel"])
        if requirements:
            run(pip_command + ["install", "--no-input", *requirements])
        run(pip_command + ["check"], check=False)
        create_run_bat(project_dir, args.venv)
        create_run_sh(project_dir, args.venv)
        print("\nInstallation complete.")
    except subprocess.CalledProcessError as exc:
        print(f"\nAn installation step failed: {exc}")
    finally:
        pause_if_double_clicked()


if __name__ == "__main__":
    main()

