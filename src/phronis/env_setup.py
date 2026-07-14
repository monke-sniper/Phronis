"""Isolated workspace environment management.

Ensures phronis runs inside a compatible Python virtual environment
regardless of the system's default Python version.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .workspace import get_workspace_path


def _venv_dir():
    return os.path.join(get_workspace_path(), ".venv")


def _venv_python():
    return os.path.join(_venv_dir(), "Scripts", "python.exe")


def _venv_pip():
    return os.path.join(_venv_dir(), "Scripts", "pip.exe")


def _venv_cli():
    return os.path.join(_venv_dir(), "Scripts", "llamafactory-cli.exe")


def _venv_phronis():
    return os.path.join(_venv_dir(), "Scripts", "phronis.exe")


def is_inside_isolated_venv():
    """Return True if the current interpreter lives inside the workspace venv."""
    return _venv_dir() in sys.executable


def _project_root_for_editable_install() -> str | None:
    """Return the repo root if this package was installed in editable mode,
    or if the repo can be discovered from the current working directory."""
    try:
        import phronis as _pkg  # Import from inside the package to get __file__ path

        inside_src = Path(_pkg.__file__).resolve().parent  # .../src/phronis
        repo_root = inside_src.parent.parent  # Go up to repo root
        if (repo_root / "pyproject.toml").is_file():
            return str(repo_root)
    except Exception:
        pass

    # Fallback: search from cwd upwards for a repo containing src/phronis
    try:
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            if (parent / "pyproject.toml").is_file() and (parent / "src" / "phronis").is_dir():
                return str(parent)
    except Exception:
        pass
    return None


def _venv_has_package(venv_py: str, package_name: str) -> bool:
    """Return True if `package_name` is importable inside the venv."""
    if not os.path.isfile(venv_py):
        return False
    try:
        result = subprocess.run(
            [venv_py, "-c", f"import {package_name}"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _current_python_info():
    """Return (major, minor, micro) for the current interpreter."""
    return sys.version_info[:3]


def is_python_version_compatible(major: int, minor: int):
    """Return True if the given Python version supports CUDA torch wheels."""
    return major == 3 and 11 <= minor <= 13


def _is_torch_compatible(python_exe: str):
    """Return True if torch is installed and has CUDA support."""
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == "CUDA"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _venv_has_cpu_torch(venv_py: str) -> bool:
    """Return True if the venv has torch installed but it's CPU-only."""
    try:
        result = subprocess.run(
            [venv_py, "-c", "import torch; print('CPU' if torch.version.cuda is None else 'CUDA')"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip() == "CPU"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return False


def _machine_has_gpu() -> bool:
    """Return True if the machine appears to have an NVIDIA GPU."""
    return shutil.which("nvidia-smi") is not None


def _pip_works(python_exe: str) -> bool:
    """Return True if `python -m pip --version` succeeds."""
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _try_bootstrap_pip(python_exe: str) -> bool:
    """Try to bootstrap pip via ensurepip.  Returns True if pip works afterwards."""
    try:
        subprocess.run(
            [python_exe, "-m", "ensurepip", "--default-pip"],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return _pip_works(python_exe)


def _find_compatible_python():
    """Find a Python 3.11–3.13 executable that supports CUDA wheels.

    1. Check current interpreter first.
    2. If current is missing CUDA, but version is OK, return it anyway
       (torch just needs reinstall).
    3. Otherwise try the Windows `py` launcher for 3.13/3.12/3.11.
    4. Verify the candidate has a working pip; skip it if not.
    """
    current = sys.executable
    cur_major, cur_minor, _ = _current_python_info()

    if is_python_version_compatible(cur_major, cur_minor):
        if _pip_works(current):
            return current
        # Current interpreter is compatible but pip is missing — try to fix.
        if _try_bootstrap_pip(current):
            return current

    # Current version is too new (or lacks pip).  Search via py launcher.
    for minor in (13, 12, 11):
        try:
            result = subprocess.run(
                ["py", f"-3.{minor}", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    if _pip_works(path):
                        return path
                    # Found interpreter but pip is broken — try to fix.
                    if _try_bootstrap_pip(path):
                        return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None


def _repair_pip_in_venv(console: Console, venv_py: str) -> None:
    """Ensure pip actually works inside the venv.

    Handles the common Windows case where pip-*.dist-info exists but the
    actual pip package directory is missing.  ensurepip sees the metadata
    and reports "already satisfied" without installing anything.
    """
    # First attempt — plain ensurepip.
    try:
        subprocess.run(
            [venv_py, "-m", "ensurepip", "--upgrade"],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Check if pip actually works.
    try:
        result = subprocess.run(
            [venv_py, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return  # pip is fine
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # pip is broken — look for orphaned dist-info and remove it so
    # ensurepip will actually reinstall.
    try:
        site_packages = subprocess.run(
            [venv_py, "-c",
             "import site; print(site.getsitepackages()[0])"],
            capture_output=True, text=True, timeout=10,
        )
        if site_packages.returncode != 0:
            return
        sp_dir = site_packages.stdout.strip()
        for entry in os.listdir(sp_dir):
            if entry.startswith("pip-") and entry.endswith(".dist-info"):
                info_path = os.path.join(sp_dir, entry)
                shutil.rmtree(info_path, ignore_errors=True)
                console.print(f"[dim]Removed orphaned {entry}[/]")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Second attempt — re-run ensurepip after cleanup.
    try:
        subprocess.run(
            [venv_py, "-m", "ensurepip", "--upgrade"],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def ensure_isolated_venv(console: Console) -> bool:
    """Create the workspace venv and install dependencies.

    Returns True on success, False otherwise.  On success the venv
    is ready to run `phronis` and `llamafactory-cli train`.
    """
    venv_dir = _venv_dir()
    venv_py = _venv_python()
    venv_pip = _venv_pip()

    # Phase 1 — create venv if it doesn't exist yet.
    if not os.path.isfile(venv_py):
        py_exe = _find_compatible_python()
        if not py_exe:
            console.print(
                "[red]No compatible Python interpreter found.[/]"
            )
            console.print(
                "[dim]phronis needs Python 3.11–3.13 with a working pip.[/]"
            )
            console.print(
                "[dim]Install Python 3.12 from https://python.org "
                "(ensure 'pip' is checked in the installer) and re-run.[/]"
            )
            return False

        try:
            with console.status(f"[bold green]Creating isolated environment ({py_exe})...", spinner="dots"):
                subprocess.run([py_exe, "-m", "venv", venv_dir], check=True)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Failed to create venv: {exc}[/]")
            return False

    # Phase 2 — verify required packages are present.
    missing_pkgs = []
    if not _venv_has_package(venv_py, "phronis"):
        missing_pkgs.append("phronis")
    if not _venv_has_package(venv_py, "llamafactory"):
        missing_pkgs.append("llamafactory")

    if not missing_pkgs:
        return True

    # Before installing anything, make sure pip actually works in the venv.
    # Some Windows venvs ship a broken pip.exe wrapper or have orphaned
    # dist-info without the actual pip package.
    _repair_pip_in_venv(console, venv_py)

    try:
        pip_check = subprocess.run(
            [venv_py, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pip_check = None

    if pip_check is None or pip_check.returncode != 0:
        console.print(
            "[red]pip is not working inside the isolated venv.[/]"
        )
        console.print(
            "[dim]Try deleting the venv and re-running: "
            f"Remove-Item -Recurse -Force \"{_venv_dir()}\"[/]"
        )
        return False

    console.print(
        f"[dim]Repairing isolated environment (missing: {', '.join(missing_pkgs)})...[/]"
    )

    # Upgrade pip using python -m pip (more reliable than pip.exe wrapper)
    try:
        with console.status("[bold green]Upgrading pip...", spinner="dots"):
            subprocess.run(
                [venv_py, "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True, text=True, timeout=120,
            )
    except subprocess.TimeoutExpired:
        console.print("[yellow]Warning: pip upgrade timed out.[/]")

    # Install torch with CUDA
    try:
        with console.status("[bold green]Installing CUDA PyTorch... (this may take several minutes)", spinner="dots"):
            subprocess.run(
                [
                    venv_py, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ],
                check=True, timeout=900,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print(
            "[red]Failed to install CUDA PyTorch. Check your internet.[/]"
        )
        return False

    # Install phronis package + LLaMA-Factory
    repo_root = _project_root_for_editable_install()
    try:
        with console.status("[bold green]Installing phronis into isolated environment...", spinner="dots"):
            if repo_root:
                subprocess.run(
                    [venv_py, "-m", "pip", "install", "-e", repo_root],
                    check=True, timeout=300,
                )
            else:
                subprocess.run(
                    [venv_py, "-m", "pip", "install", "phronis"],
                    check=True, timeout=300,
                )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        console.print(
            "[red]Failed to install phronis into isolated environment.[/]"
        )
        return False

    # Re-verify after install
    if not _venv_has_package(venv_py, "phronis"):
        console.print(
            "[red]phronis still not importable after installation. "
            "Check the pip output above for errors.[/]"
        )
        return False

    _create_wrapper_script(console, venv_dir)

    # Phase 4 — verify venv torch is CUDA-capable on GPU machines.
    # Installing phronis/llamafactory can clobber a CUDA torch with CPU-only wheels.
    if _venv_has_cpu_torch(venv_py):
        if _machine_has_gpu():
            console.print(
                "[yellow]Workspace venv has CPU-only PyTorch but an NVIDIA GPU was detected. "
                "Reinstalling CUDA version...[/]"
            )
            try:
                subprocess.run(
                    [
                        venv_py, "-m", "pip", "install", "--force-reinstall",
                        "torch", "torchvision", "torchaudio",
                        "--index-url", "https://download.pytorch.org/whl/cu124",
                    ],
                    check=True, timeout=900,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                console.print(
                    "[red]Failed to reinstall CUDA PyTorch. GPU training will be very slow.[/]"
                )
            else:
                if _venv_has_cpu_torch(venv_py):
                    console.print(
                        "[red]Torch is still CPU-only after reinstall. "
                        "Check your GPU drivers.[/]"
                    )
        else:
            console.print("[dim]No GPU detected; CPU-only PyTorch is correct.[/]")

    return True


def _create_wrapper_script(console: Console, venv_dir: str):
    """Write a tiny launcher so users can add it to their PATH."""
    wrapper = os.path.join(get_workspace_path(), "phronis.cmd")
    venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    cmd_body = (
        "@echo off\n"
        f'"{venv_py}" -m phronis %*\n'
    )
    try:
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(cmd_body)
    except OSError:
        pass
    else:
        console.print(
            f"[dim]Launcher written to:[/] [bold]{wrapper}[/]"
        )
        console.print(
            "[dim]Add that folder to your PATH for a quicker `phronis` command.[/]"
        )


def forward_to_venv(argv=None):
    """Re-execute the current command using the isolated venv interpreter.

    Returns None if the venv is missing or phronis is not installed inside it;
    otherwise returns the CompletedProcess from the forwarded run.
    """
    if argv is None:
        argv = sys.argv
    venv_py = _venv_python()
    if not os.path.isfile(venv_py):
        return None
    if not _venv_has_package(venv_py, "phronis"):
        return None
    return subprocess.run([venv_py, "-m", "phronis"] + argv[1:])
