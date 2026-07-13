
import os
import yaml

DEFAULT_WORKSPACE = os.path.join(os.path.expanduser("~"), ".phronisworkspace")
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".phronisworkspace", "workspace.yaml")

GUIDE_CONTENT = """# phronis Guide

## Workspace
Your workspace is the central location for all phronis output files (saves, models, configs).
Default: ~/.phronisworkspace/

### Structure
    ~/.phronisworkspace/
    ├── workspace.yaml      # workspace config
    ├── .phronis.yaml      # app state
    ├── saves/              # LoRA adapters + checkpoints
    ├── models/             # exported / merged models
    └── configs/            # auto-generated YAML from training runs

## Central Data Directories (repo-level)
    <repo_root>/
    ├── data/               # ONE central data folder — drop datasets here
    │   ├── dataset_info.json
    │   └── README.txt
    └── yaml/               # saved YAML training configs

## How to Change Workspace Location
Edit workspace.yaml:
    workspace_path: /your/custom/path

Or set environment variable:
    PHRONIS_WORKSPACE=/your/custom/path

## Menu Options

### 1. Quick Train
Fine-tune in 3 prompts. Picks model, dataset, epochs. Uses smart defaults.
Supports optional target-loss training.

### 2. Advanced Training
Full control over all hyperparameters plus optional target-loss training.

### 3. Train from YAML
Select a saved YAML config from the central yaml/ folder and train directly.

### 4. Chat Trained Model
Instantly chat with your last fine-tune.

### 5. Quick Chat
Chat with any cached model.

### 6. Download Model
Search and download models from HuggingFace. Shows file sizes and progress.

### 7. Download Dataset
Search and download datasets from HuggingFace. Auto-detects format.

### 8. Export Adapter
Merge LoRA adapter into a standalone model.

### 9. View Models
Browse your cached models. Set active model.

### 10. View Datasets
Browse available datasets. Set active dataset.

### 11. Add Dataset
Register a dataset manually (local file or HuggingFace URL).

### 12. Workspace Info
Show workspace location, directory sizes, and file counts.

### 13. System Check
Verify Python, LLaMA-Factory, GPU, and workspace setup.

## Dataset Formats

### Alpaca (.json)
    [
      {"instruction": "What is the capital of France?", "output": "Paris"}
    ]

### ShareGPT (.json or .jsonl)
    [
      {
        "messages": [
          {"role": "user", "content": "Hello!"},
          {"role": "assistant", "content": "Hi there!"}
        ]
      }
    ]

### Auto-Detection
Drop .json or .jsonl files in data/ and they appear automatically.
Supported patterns: instruction/output, messages, conversations, prompt/completion, text.

## Training Configs
Each training run produces a YAML config in configs/.
You can manually edit these YAMLs to customize training.
The configs are modular - each run has its own file.

Saved YAML configs can be placed in yaml/ and selected via "Train from YAML".

## CLI Commands

### Interactive
    phronis                              # launch interactive menu
    phronis --version                    # show version
    phronis --help                       # show all commands

### Training
    phronis train --model X --dataset Y  # direct training
    phronis train --model X --dataset Y --dry-run    # print config, don't train
    phronis train --model X --dataset Y --resume /path/to/checkpoint
    phronis train --model X --dataset Y --method lora --grad-accum 8
    phronis train --model X --dataset Y --scheduler cosine --warmup 0.1
    phronis train --model X --dataset Y --force          # overwrite output dir
    phronis train --model X --dataset Y --push-to-hub    # upload after training
    phronis train --model X --dataset Y --target-loss 0.9  # stop at loss ~0.9
    phronis yaml-train /path/to/config.yaml             # train from YAML file

### Chat
    phronis chat                         # interactive chat with active model
    phronis chat --model X --message "Hello"          # single-shot response
    phronis chat --model X --adapter Y --message "Hi"   # chat with adapter

### Export
    phronis export --adapter saves/run/lora           # merge adapter
    phronis export --adapter saves/run/lora --output models/my_model

### Download
    phronis download model Qwen/Qwen3-0.6B             # download a model
    phronis download dataset tatsu-lab/alpaca          # download a dataset
    phronis download model Qwen/Qwen3-0.6B --no-confirm # skip confirmation

### List
    phronis list models                  # list cached models
    phronis list datasets                # list available datasets
    phronis list history                 # list training runs
    phronis list adapters                # list LoRA adapters
    phronis list models --json           # output as JSON

### Add Dataset
    phronis add dataset --name my_data --file my_data.json --format alpaca
    phronis add dataset --name my_data --hf-url https://huggingface.co/datasets/... --format sharegpt

### Workspace
    phronis info                         # show workspace info and disk usage
    phronis info --json                    # output as JSON
    phronis doctor                       # full system diagnostic
    phronis doctor --fix                 # auto-fix missing dependencies
    phronis clean configs                # delete old config files
    phronis clean all --force            # delete everything (no confirmation)

### Maintenance
    phronis setup                        # run setup/health check
    phronis update                       # self-update via pip
    phronis update --check               # check for updates only

### Global Flags
    phronis --workspace /path/to/custom  # override workspace directory
    phronis --no-color                   # disable colored output
    phronis --quiet                      # suppress non-essential output
    phronis --verbose                    # verbose/debug output

## Environment Variables
    PHRONIS_WORKSPACE   # override workspace location
    HF_HOME               # HuggingFace cache location
"""


def get_workspace_path() -> str:
    if env_path := os.environ.get("PHRONIS_WORKSPACE"):
        return os.path.abspath(env_path)

    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if path := cfg.get("workspace_path"):
                return os.path.abspath(path)
        except (yaml.YAMLError, OSError):
            pass

    return DEFAULT_WORKSPACE


def set_workspace_path(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    config = {"workspace_path": path}
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return path


def _compute_workspace_dirs(workspace_path):
    return {
        "saves": os.path.join(workspace_path, "saves"),
        "models": os.path.join(workspace_path, "models"),
        "configs": os.path.join(workspace_path, "configs"),
    }


def ensure_workspace_dirs(workspace_path: str) -> dict[str, str]:
    dirs = _compute_workspace_dirs(workspace_path)
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def write_workspace_guide(workspace_path: str) -> None:
    guide_path = os.path.join(workspace_path, "GUIDE.md")
    if not os.path.isfile(guide_path):
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(GUIDE_CONTENT)


def init_workspace() -> tuple[str, dict[str, str]]:
    workspace_path = get_workspace_path()
    dirs = ensure_workspace_dirs(workspace_path)
    write_workspace_guide(workspace_path)
    return workspace_path, dirs


def sync_demo_datasets(bundled_data_dir: str, bundled_dataset_info: str, data_dir: str, dataset_info_path: str) -> None:
    """No-op: data is now centrally managed in the repo root data/ folder."""
    pass
