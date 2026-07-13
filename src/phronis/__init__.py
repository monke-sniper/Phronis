import os

from .workspace import get_workspace_path, _compute_workspace_dirs

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels above src/phronis
REPO_ROOT = os.path.dirname(os.path.dirname(_pkg_dir))

# Output directories (workspace-based) — computed without I/O side effects
_workspace_path = get_workspace_path()
_dirs = _compute_workspace_dirs(_workspace_path)
PROJECT_ROOT = _workspace_path
STATE_PATH = os.path.join(PROJECT_ROOT, ".phronis.yaml")
SAVES_DIR = _dirs["saves"]
MODELS_DIR = _dirs["models"]
CONFIGS_DIR = _dirs["configs"]
DATA_DIR = _dirs["data"]
YAML_DIR = _dirs["yaml"]

# Dataset registry lives in the workspace data dir
DATASET_INFO = os.path.join(DATA_DIR, "dataset_info.json")
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

# Source of bundled demo datasets inside the package
BUNDLED_DATA_SOURCE = os.path.join(_pkg_dir, "data")

# Backward-compat aliases
BUNDLED_DATA_DIR = DATA_DIR
BUNDLED_DATASET_INFO = DATASET_INFO

DEFAULT_CONFIG = {
    "active_model": "",
    "active_adapter": "",
    "active_template": "qwen3",
    "active_dataset": "",
    "training_history": [],
    "theme": "dark",
}
