import io
import os
import subprocess
import sys
import tempfile

import pytest
from rich.console import Console


def _dummy_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


class TestIsPythonVersionCompatible:
    def test_311_compatible(self):
        from phronis.env_setup import is_python_version_compatible
        assert is_python_version_compatible(3, 11)

    def test_312_compatible(self):
        from phronis.env_setup import is_python_version_compatible
        assert is_python_version_compatible(3, 12)

    def test_313_compatible(self):
        from phronis.env_setup import is_python_version_compatible
        assert is_python_version_compatible(3, 13)

    def test_314_incompatible(self):
        from phronis.env_setup import is_python_version_compatible
        assert not is_python_version_compatible(3, 14)

    def test_310_incompatible(self):
        from phronis.env_setup import is_python_version_compatible
        assert not is_python_version_compatible(3, 10)


class TestIsInsideIsolatedVenv:
    def test_true_when_executable_inside_venv(self, monkeypatch):
        from phronis.env_setup import is_inside_isolated_venv

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "phronis.env_setup.get_workspace_path", lambda: tmp
            )
            fake_py = os.path.join(tmp, ".venv", "Scripts", "python.exe")
            old = sys.executable
            try:
                monkeypatch.setattr(sys, "executable", fake_py)
                assert is_inside_isolated_venv()
            finally:
                monkeypatch.setattr(sys, "executable", old)

    def test_false_when_outside_venv(self, monkeypatch):
        from phronis.env_setup import is_inside_isolated_venv

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "phronis.env_setup.get_workspace_path", lambda: tmp
            )
            assert not is_inside_isolated_venv()


class TestEnsureIsolatedVenv:
    def test_reuses_existing_venv(self, monkeypatch):
        import phronis.env_setup as es
        from phronis.env_setup import ensure_isolated_venv

        with tempfile.TemporaryDirectory() as tmp:
            venv_dir = os.path.join(tmp, ".venv")
            scripts = os.path.join(venv_dir, "Scripts")
            os.makedirs(scripts, exist_ok=True)
            open(os.path.join(scripts, "python.exe"), "w").close()
            open(os.path.join(scripts, "pip.exe"), "w").close()

            monkeypatch.setattr(
                "phronis.env_setup.get_workspace_path", lambda: tmp
            )
            # Should short-circuit without calling any subprocesses
            assert ensure_isolated_venv(_dummy_console()) is True

    def test_fails_when_no_compatible_python(self, monkeypatch):
        import phronis.env_setup as es
        from phronis.env_setup import ensure_isolated_venv

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "phronis.env_setup.get_workspace_path", lambda: tmp
            )
            monkeypatch.setattr(es, "is_python_version_compatible", lambda m, n: False)
            monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1))

            assert ensure_isolated_venv(_dummy_console()) is False


class TestForwardToVenv:
    def test_returns_none_when_venv_missing(self, monkeypatch):
        from phronis.env_setup import forward_to_venv
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "phronis.env_setup.get_workspace_path", lambda: tmp
            )
            result = forward_to_venv(["phronis", "--version"])
            assert result is False


class TestBootstrapInstallMissing:
    def test_torch_install_includes_cuda_index(self, monkeypatch):
        from phronis.bootstrap import _install_missing
        import phronis.bootstrap as boot

        captures = []
        original_run = subprocess.run

        def _capture(*args, **kwargs):
            captures.append(args)
            return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _capture)

        console = _dummy_console()
        _install_missing(console, ["PyTorch"])

        assert any(
            "https://download.pytorch.org/whl/cu124" in str(arg)
            for arg in captures
        ), f"Expected CUDA index in pip args, got: {captures}"
