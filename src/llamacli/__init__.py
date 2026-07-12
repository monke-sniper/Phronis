import os

from .workspace import init_workspace

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels above src/llamacli
REPO_ROOT = os.path.dirname(os.path.dirname(_pkg_dir))

# Central input directories (repo-level, shared across workspaces)
DATA_DIR = os.path.join(REPO_ROOT, "data")
YAML_DIR = os.path.join(REPO_ROOT, "yaml")

# Output directories (workspace-based)
_workspace_path, _dirs = init_workspace()
PROJECT_ROOT = _workspace_path
STATE_PATH = os.path.join(PROJECT_ROOT, ".llamacli.yaml")
SAVES_DIR = _dirs["saves"]
MODELS_DIR = _dirs["models"]
CONFIGS_DIR = _dirs["configs"]

# Dataset registry lives in the central data dir
DATASET_INFO = os.path.join(DATA_DIR, "dataset_info.json")
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

# Backward-compat aliases (central data is the bundled data)
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
