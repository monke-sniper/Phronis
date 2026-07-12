import io
import os
import shutil
import tempfile

import pytest
from rich.console import Console


@pytest.fixture
def mock_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


@pytest.fixture
def temp_workspace(monkeypatch):
    """Provide an isolated workspace directory and patch phronis paths."""
    tmp = tempfile.mkdtemp()
    import phronis
    import phronis.cli as cli_mod
    import phronis.prompts as prompts_mod
    import phronis.state as state_mod

    old = {
        "PROJECT_ROOT": phronis.PROJECT_ROOT,
        "DATA_DIR": phronis.DATA_DIR,
        "SAVES_DIR": phronis.SAVES_DIR,
        "MODELS_DIR": phronis.MODELS_DIR,
        "CONFIGS_DIR": phronis.CONFIGS_DIR,
        "DATASET_INFO": phronis.DATASET_INFO,
        "STATE_PATH": state_mod.STATE_PATH,
    }

    root = tmp
    data = os.path.join(root, "data")
    saves = os.path.join(root, "saves")
    models = os.path.join(root, "models")
    configs = os.path.join(root, "configs")
    dsi = os.path.join(data, "dataset_info.json")
    state_path = os.path.join(root, ".phronis.yaml")

    for d in (data, saves, models, configs):
        os.makedirs(d, exist_ok=True)

    phronis.PROJECT_ROOT = root
    phronis.DATA_DIR = data
    phronis.SAVES_DIR = saves
    phronis.MODELS_DIR = models
    phronis.CONFIGS_DIR = configs
    phronis.DATASET_INFO = dsi
    state_mod.STATE_PATH = state_path

    path_map = {
        "PROJECT_ROOT": root,
        "DATA_DIR": data,
        "SAVES_DIR": saves,
        "MODELS_DIR": models,
        "CONFIGS_DIR": configs,
        "DATASET_INFO": dsi,
    }
    for mod in (cli_mod, prompts_mod):
        for k, v in path_map.items():
            if hasattr(mod, k):
                setattr(mod, k, v)

    state_mod._state = None

    yield tmp

    # Restore
    state_mod.STATE_PATH = old["STATE_PATH"]
    state_mod._state = None
    for k, v in old.items():
        if k == "STATE_PATH":
            continue
        setattr(phronis, k, v)
    for mod in (cli_mod, prompts_mod):
        for k, v in old.items():
            if hasattr(mod, k):
                setattr(mod, k, v)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
