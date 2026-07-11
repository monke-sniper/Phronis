import json
import os
import tempfile

import pytest


class TestSyncDemoDatasets:
    def test_copies_missing_files_and_registers(self):
        from llamacli.workspace import sync_demo_datasets

        with tempfile.TemporaryDirectory() as tmp:
            bundled_dir = os.path.join(tmp, "bundled")
            os.makedirs(bundled_dir)
            data_dir = os.path.join(tmp, "workspace_data")
            os.makedirs(data_dir)
            bundled_info = os.path.join(bundled_dir, "dataset_info.json")
            workspace_info = os.path.join(data_dir, "dataset_info.json")

            with open(bundled_info, "w", encoding="utf-8") as f:
                json.dump(
                    {"identity": {"file_name": "identity.json", "formatting": "alpaca"}},
                    f,
                )
            with open(
                os.path.join(bundled_dir, "identity.json"), "w", encoding="utf-8"
            ) as f:
                json.dump([{"instruction": "hi", "output": "hello"}], f)

            sync_demo_datasets(bundled_dir, bundled_info, data_dir, workspace_info)

            assert os.path.isfile(os.path.join(data_dir, "identity.json"))
            with open(workspace_info, "r", encoding="utf-8") as f:
                registry = json.load(f)
            assert "identity" in registry
            assert registry["identity"]["file_name"] == "identity.json"

    def test_does_not_overwrite_newer_workspace_file(self):
        from llamacli.workspace import sync_demo_datasets

        with tempfile.TemporaryDirectory() as tmp:
            bundled_dir = os.path.join(tmp, "bundled")
            os.makedirs(bundled_dir)
            data_dir = os.path.join(tmp, "workspace_data")
            os.makedirs(data_dir)
            bundled_info = os.path.join(bundled_dir, "dataset_info.json")
            workspace_info = os.path.join(data_dir, "dataset_info.json")

            with open(bundled_info, "w", encoding="utf-8") as f:
                json.dump(
                    {"identity": {"file_name": "identity.json", "formatting": "alpaca"}},
                    f,
                )
            with open(
                os.path.join(bundled_dir, "identity.json"), "w", encoding="utf-8"
            ) as f:
                json.dump([{"instruction": "old"}], f)

            # Create older workspace file
            with open(os.path.join(data_dir, "identity.json"), "w", encoding="utf-8") as f:
                json.dump([{"instruction": "newer"}], f)
            # Touch bundled file to be older
            os.utime(
                os.path.join(bundled_dir, "identity.json"),
                times=(0, 0),
            )

            sync_demo_datasets(bundled_dir, bundled_info, data_dir, workspace_info)

            with open(os.path.join(data_dir, "identity.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data == [{"instruction": "newer"}]

    def test_skips_when_bundled_dir_missing(self):
        from llamacli.workspace import sync_demo_datasets

        with tempfile.TemporaryDirectory() as tmp:
            sync_demo_datasets(
                os.path.join(tmp, "nonexistent"),
                os.path.join(tmp, "nonexistent.json"),
                tmp,
                os.path.join(tmp, "info.json"),
            )
            assert not os.path.isfile(os.path.join(tmp, "info.json"))
