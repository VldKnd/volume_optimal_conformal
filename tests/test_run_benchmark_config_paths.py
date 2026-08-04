from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_benchmark import _discover_config_paths


class BenchmarkConfigPathDiscoveryTest(unittest.TestCase):

    def test_accepts_one_yaml_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "seed_00.yaml"
            config_path.write_text("name: seed_00\n", encoding="utf-8")

            self.assertEqual(_discover_config_paths(config_path), [config_path])

    def test_discovers_yaml_files_recursively_and_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            first = root / "seed_00.yml"
            second = nested / "seed_01.yaml"
            ignored = nested / "notes.txt"
            first.write_text("name: seed_00\n", encoding="utf-8")
            second.write_text("name: seed_01\n", encoding="utf-8")
            ignored.write_text("not a config\n", encoding="utf-8")

            self.assertEqual(
                _discover_config_paths(root),
                sorted([first, second]),
            )

    def test_rejects_a_non_yaml_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must have a .yaml or .yml"):
                _discover_config_paths(config_path)

    def test_rejects_an_empty_directory_and_a_missing_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            with self.assertRaisesRegex(FileNotFoundError, "No YAML"):
                _discover_config_paths(root)

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                _discover_config_paths(root / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
