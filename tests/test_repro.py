import io
import os
import tempfile

import yaml
from rich.console import Console
from typer.testing import CliRunner


def _dummy_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


class TestWriteConfigAndTrain:
    def test_creates_yaml_with_repro_header(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            import phronis.cli as cli_mod

            old_configs = phronis.CONFIGS_DIR
            phronis.CONFIGS_DIR = tmp
            cli_mod.CONFIGS_DIR = tmp

            monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: True)

            try:
                config = {
                    "model_name_or_path": "test/model",
                    "dataset": "identity",
                    "template": "qwen3",
                }
                cli_mod._write_config_and_train(
                    _dummy_console(),
                    config,
                    "test_run",
                    command="quick_train",
                    model="test/model",
                    dataset="identity",
                    epochs=3,
                )
                path = os.path.join(tmp, "phronis_test_run.yaml")
                assert os.path.isfile(path)

                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                assert "phronis reproducibility manifest" in raw
                assert "command: quick_train" in raw
                assert "random_seed:" in raw

                loaded = yaml.safe_load(raw)
                assert loaded["model_name_or_path"] == "test/model"
                assert "seed" in loaded
                assert isinstance(loaded["seed"], int)
                assert loaded["output_dir"] == os.path.join("saves", "test_run", "lora")
            finally:
                phronis.CONFIGS_DIR = old_configs
                cli_mod.CONFIGS_DIR = old_configs

    def test_preserves_existing_seed(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            import phronis.cli as cli_mod

            old_configs = phronis.CONFIGS_DIR
            phronis.CONFIGS_DIR = tmp
            cli_mod.CONFIGS_DIR = tmp

            monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: True)

            try:
                config = {"seed": 42}
                cli_mod._write_config_and_train(
                    _dummy_console(), config, "seeded_run",
                    command="train", model="m", dataset="d",
                )
                path = os.path.join(tmp, "phronis_seeded_run.yaml")
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                assert loaded["seed"] == 42
            finally:
                phronis.CONFIGS_DIR = old_configs
                cli_mod.CONFIGS_DIR = old_configs

    def test_output_dir_matches_run_name(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            import phronis.cli as cli_mod

            old_configs = phronis.CONFIGS_DIR
            phronis.CONFIGS_DIR = tmp
            cli_mod.CONFIGS_DIR = tmp

            monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: True)

            try:
                # quick_train historically passed an empty output_name to _build_config,
                # leaving output_dir as "saves//lora". _write_config_and_train must fix it.
                config = {
                    "model_name_or_path": "m",
                    "output_dir": "saves//lora",
                }
                cli_mod._write_config_and_train(
                    _dummy_console(), config, "fixed_run",
                    command="quick_train", model="m", dataset="d",
                )
                path = os.path.join(tmp, "phronis_fixed_run.yaml")
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                assert loaded["output_dir"] == os.path.join("saves", "fixed_run", "lora")
            finally:
                phronis.CONFIGS_DIR = old_configs
                cli_mod.CONFIGS_DIR = old_configs


class TestTyperTrainCommand:
    def test_cli_train_writes_yaml(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            import phronis
            import phronis.cli as cli_mod

            old_configs = phronis.CONFIGS_DIR
            phronis.CONFIGS_DIR = tmp
            cli_mod.CONFIGS_DIR = tmp

            monkeypatch.setattr(cli_mod, "run_training", lambda c, p, o, **kw: True)
            monkeypatch.setattr(cli_mod, "_record_training", lambda *a, **k: None)

            runner = CliRunner()
            try:
                result = runner.invoke(
                    cli_mod.app,
                    [
                        "train",
                        "--model", "Qwen/Qwen3-0.6B",
                        "--dataset", "identity",
                        "-o", "cli_test",
                    ],
                )
                assert result.exit_code == 0
                path = os.path.join(tmp, "phronis_cli_test.yaml")
                assert os.path.isfile(path)

                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                assert "command: train" in raw
                assert "cli_args.model: Qwen/Qwen3-0.6B" in raw

                loaded = yaml.safe_load(raw)
                assert loaded["dataset"] == "identity"
                assert "seed" in loaded
            finally:
                phronis.CONFIGS_DIR = old_configs
                cli_mod.CONFIGS_DIR = old_configs
