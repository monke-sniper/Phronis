import shutil

import json
import os
import yaml

DEFAULT_WORKSPACE = os.path.join(os.path.expanduser("~"), ".llamaworkspace")
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".llamaworkspace", "workspace.yaml")

GUIDE_CONTENT = """# llamacli Guide

## Workspace
Your workspace is the central location for all llamacli output files (saves, models, configs).
Default: ~/.llamaworkspace/

### Structure
    ~/.llamaworkspace/
    ├── workspace.yaml      # workspace config
    ├── .llamacli.yaml      # app state
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
    LLAMACLII_WORKSPACE=/your/custom/path

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
    llamacli                              # launch interactive menu
    llamacli --version                    # show version
    llamacli --help                       # show all commands

### Training
    llamacli train --model X --dataset Y  # direct training
    llamacli train --model X --dataset Y --dry-run    # print config, don't train
    llamacli train --model X --dataset Y --resume /path/to/checkpoint
    llamacli train --model X --dataset Y --method lora --grad-accum 8
    llamacli train --model X --dataset Y --scheduler cosine --warmup 0.1
    llamacli train --model X --dataset Y --force          # overwrite output dir
    llamacli train --model X --dataset Y --push-to-hub    # upload after training
    llamacli train --model X --dataset Y --target-loss 0.9  # stop at loss ~0.9
    llamacli yaml-train /path/to/config.yaml             # train from YAML file

### Chat
    llamacli chat                         # interactive chat with active model
    llamacli chat --model X --message "Hello"          # single-shot response
    llamacli chat --model X --adapter Y --message "Hi"   # chat with adapter

### Export
    llamacli export --adapter saves/run/lora           # merge adapter
    llamacli export --adapter saves/run/lora --output models/my_model

### Download
    llamacli download model Qwen/Qwen3-0.6B             # download a model
    llamacli download dataset tatsu-lab/alpaca          # download a dataset
    llamacli download model Qwen/Qwen3-0.6B --no-confirm # skip confirmation

### List
    llamacli list models                  # list cached models
    llamacli list datasets                # list available datasets
    llamacli list history                 # list training runs
    llamacli list adapters                # list LoRA adapters
    llamacli list models --json           # output as JSON

### Add Dataset
    llamacli add dataset --name my_data --file my_data.json --format alpaca
    llamacli add dataset --name my_data --hf-url https://huggingface.co/datasets/... --format sharegpt

### Workspace
    llamacli info                         # show workspace info and disk usage
    llamacli info --json                    # output as JSON
    llamacli doctor                       # full system diagnostic
    llamacli doctor --fix                 # auto-fix missing dependencies
    llamacli clean configs                # delete old config files
    llamacli clean all --force            # delete everything (no confirmation)

### Maintenance
    llamacli setup                        # run setup/health check
    llamacli update                       # self-update via pip
    llamacli update --check               # check for updates only

### Global Flags
    llamacli --workspace /path/to/custom  # override workspace directory
    llamacli --no-color                   # disable colored output
    llamacli --quiet                      # suppress non-essential output
    llamacli --verbose                    # verbose/debug output

## Environment Variables
    LLAMACLII_WORKSPACE   # override workspace location
    HF_HOME               # HuggingFace cache location
"""


def get_workspace_path():
    if env_path := os.environ.get("LLAMACLII_WORKSPACE"):
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


def set_workspace_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    config = {"workspace_path": path}
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return path


def ensure_workspace_dirs(workspace_path):
    dirs = {
        "saves": os.path.join(workspace_path, "saves"),
        "models": os.path.join(workspace_path, "models"),
        "configs": os.path.join(workspace_path, "configs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    guide_path = os.path.join(workspace_path, "GUIDE.md")
    if not os.path.isfile(guide_path):
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(GUIDE_CONTENT)

    return dirs


def init_workspace():
    workspace_path = get_workspace_path()
    dirs = ensure_workspace_dirs(workspace_path)
    return workspace_path, dirs


def sync_demo_datasets(bundled_data_dir, bundled_dataset_info, data_dir, dataset_info_path):
    """No-op: data is now centrally managed in the repo root data/ folder."""
    pass
