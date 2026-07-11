import os
import random
import subprocess
import sys
from datetime import datetime, timezone


def _get_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .strip()
        )
    except Exception:
        return ""


def _get_llamacli_version():
    try:
        from importlib.metadata import version

        return version("llamacli")
    except Exception:
        return "0.1.0"


def _get_llamafactory_version():
    try:
        import llamafactory

        return getattr(llamafactory, "__version__", "unknown")
    except Exception:
        return "not_installed"


def gather_repro_metadata(command: str, **kwargs) -> dict:
    """Return a reproducibility metadata dict for a training run.

    Parameters
    ----------
    command : str
        The command route that triggered the run, e.g. 'quick_train',
        'advanced_train', or 'train'.
    **kwargs
        Extra args to record (e.g. model, dataset, epochs, etc.).
    """
    seed = kwargs.get("seed", random.randint(0, 2**32 - 1))
    return {
        "llamacli_version": _get_llamacli_version(),
        "llamafactory_version": _get_llamafactory_version(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "git_commit": _get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cli_args": kwargs,
        "random_seed": seed,
    }


def format_repro_header(metadata: dict) -> str:
    """Serialize metadata as a YAML comment block header.

    This keeps the metadata readable while ensuring it does not interfere
    with LLaMA-Factory's YAML parser.
    """
    lines = ["# llamacli reproducibility manifest", "# ---------------------------------"]
    for key, value in metadata.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"#   {key}.{sub_key}: {sub_value}")
        else:
            lines.append(f"#   {key}: {value}")
    return "\n".join(lines) + "\n"
