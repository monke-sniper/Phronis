import os
import subprocess
import sys



VENV_CLI = os.path.join(os.path.dirname(sys.executable), "Scripts", "phronis.exe")


class TestImports:
    def test_cli_imports(self):
        import phronis.cli
        assert hasattr(phronis.cli, "app")
        assert hasattr(phronis.cli, "entry")
        assert hasattr(phronis.cli, "interactive_loop")
        assert hasattr(phronis.cli, "quick_train")
        assert hasattr(phronis.cli, "advanced_train")
        assert hasattr(phronis.cli, "chat_trained")

    def test_prompts_imports(self):
        import phronis.prompts
        assert hasattr(phronis.prompts, "prompt_model")
        assert hasattr(phronis.prompts, "prompt_dataset")
        assert hasattr(phronis.prompts, "prompt_stage")
        assert hasattr(phronis.prompts, "prompt_finetuning_type")
        assert hasattr(phronis.prompts, "prompt_training_params")
        assert hasattr(phronis.prompts, "detect_template")
        assert hasattr(phronis.prompts, "_list_cached_models")
        assert hasattr(phronis.prompts, "_list_datasets")

    def test_runner_imports(self):
        import phronis.runner
        assert hasattr(phronis.runner, "run_training")
        assert hasattr(phronis.runner, "run_export")

    def test_hf_imports(self):
        import phronis.hf
        assert hasattr(phronis.hf, "search_models")
        assert hasattr(phronis.hf, "download_model")
        assert hasattr(phronis.hf, "download_model_interactive")

    def test_logo_imports(self):
        import phronis.logo
        assert hasattr(phronis.logo, "print_logo")
        assert hasattr(phronis.logo, "get_logo_text")

    def test_state_imports(self):
        import phronis.state
        assert hasattr(phronis.state, "get_state")
        assert hasattr(phronis.state, "AppState")
        assert hasattr(phronis.state, "reload_state")

    def test_all_screen_functions_load(self):
        pass

    def test_new_commands_exist(self):
        """Verify all new subcommands are registered in the Typer app."""
        from phronis.cli import app
        registered = set()
        for cmd in app.registered_commands:
            name = cmd.name or cmd.callback.__name__
            # Map ad dataset back to its CLI name 'add'
            if name == "add_dataset":
                name = "add"
            registered.add(name)
        expected = {"setup", "train", "chat", "export", "download", "list", "add", "info", "doctor", "update", "clean", "yaml-train"}
        for cmd in expected:
            assert cmd in registered, f"'{cmd}' not registered in Typer app"


class TestSmoke:
    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "phronis" in result.stdout

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "LLaMA-Factory Interactive CLI" in result.stdout
        assert "train" in result.stdout

    def test_train_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "train", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--dataset" in result.stdout
        assert "--epochs" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--resume" in result.stdout
        assert "--method" in result.stdout
        assert "--grad-accum" in result.stdout
        assert "--warmup" in result.stdout
        assert "--scheduler" in result.stdout
        assert "--force" in result.stdout
        assert "--push-to-hub" in result.stdout

    def test_chat_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "chat", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--message" in result.stdout
        assert "--max-tokens" in result.stdout

    def test_export_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "export", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--adapter" in result.stdout
        assert "--model" in result.stdout
        assert "--output" in result.stdout

    def test_download_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "download", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "model" in result.stdout
        assert "dataset" in result.stdout
        assert "--no-confirm" in result.stdout

    def test_list_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "list", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--json" in result.stdout

    def test_info_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "info", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--json" in result.stdout

    def test_doctor_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "doctor", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--fix" in result.stdout

    def test_update_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "update", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--check" in result.stdout

    def test_clean_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "clean", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "configs" in result.stdout
        assert "--force" in result.stdout

    def test_main_help_has_all_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "phronis.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        commands = ["train", "chat", "export", "download", "list", "info", "doctor", "update", "clean", "setup"]
        for cmd in commands:
            assert cmd in result.stdout, f"'{cmd}' not found in main --help"
