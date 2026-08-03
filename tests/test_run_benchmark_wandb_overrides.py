from __future__ import annotations

import copy
import unittest
from argparse import Namespace

from scripts.run_benchmark import _apply_wandb_overrides


class _FakeExperimentConfig:

    def __init__(self, data):
        self.data = copy.deepcopy(data)

    def model_dump(self):
        return copy.deepcopy(self.data)

    @classmethod
    def model_validate(cls, data):
        return cls(data)


def _arguments(**overrides):
    values = {
        "wandb_mode": None,
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_group": None,
        "wandb_name": None,
        "wandb_tags": None,
    }
    values.update(overrides)
    return Namespace(**values)


class WandbCommandLineOverrideTest(unittest.TestCase):

    def test_no_overrides_returns_original_config(self):
        config = _FakeExperimentConfig({"name": "seed_00", "wandb": None})

        result = _apply_wandb_overrides(config, _arguments())

        self.assertIs(result, config)

    def test_only_explicit_values_override_yaml(self):
        config = _FakeExperimentConfig(
            {
                "name": "seed_00",
                "wandb": {
                    "mode": "offline",
                    "project": "yaml-project",
                    "group": "yaml-group",
                    "tags": ["yaml-tag"],
                },
            }
        )

        result = _apply_wandb_overrides(
            config,
            _arguments(
                wandb_mode="online",
                wandb_project="cli-project",
            ),
        )

        self.assertEqual(result.data["wandb"]["mode"], "online")
        self.assertEqual(result.data["wandb"]["project"], "cli-project")
        self.assertEqual(result.data["wandb"]["group"], "yaml-group")
        self.assertEqual(result.data["wandb"]["tags"], ["yaml-tag"])

    def test_tags_replace_yaml_tags(self):
        config = _FakeExperimentConfig(
            {
                "name": "seed_00",
                "wandb": {
                    "mode": "disabled",
                    "tags": ["old"],
                },
            }
        )

        result = _apply_wandb_overrides(
            config,
            _arguments(wandb_tags=["scm20d", "dopri5"]),
        )

        self.assertEqual(result.data["wandb"]["tags"], ["scm20d", "dopri5"])


if __name__ == "__main__":
    unittest.main()
