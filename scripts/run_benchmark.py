from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from experimentation import ExperimentRunner, load_experiment_config


def _apply_wandb_overrides(config, args):
    """Apply only explicitly provided command-line W&B settings."""
    overrides = {
        key: value
        for key, value in {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "name": args.wandb_name,
            "tags": args.wandb_tags,
        }.items() if value is not None
    }
    if not overrides:
        return config

    config_data = config.model_dump()
    wandb_data = config_data.get("wandb") or {}
    wandb_data.update(overrides)
    config_data["wandb"] = wandb_data
    return type(config).model_validate(config_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark configurations sequentially."
    )
    parser.add_argument(
        "config_directory",
        type=Path,
        help="Directory containing YAML configurations.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=None,
        help="Override W&B mode for every selected configuration.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override the W&B project for every selected configuration.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Override the W&B entity for every selected configuration.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Override the W&B group for every selected configuration.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help=(
            "Override the W&B run name for every selected configuration; "
            "prefer YAML names when running more than one configuration."
        ),
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Replace W&B tags for every selected configuration.",
    )
    args = parser.parse_args()

    config_directory = args.config_directory
    if not config_directory.is_absolute():
        config_directory = REPOSITORY_ROOT / config_directory

    config_paths = sorted(config_directory.rglob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No YAML configurations found in {config_directory}.")

    configs = [
        (
            config_path,
            _apply_wandb_overrides(load_experiment_config(config_path), args),
        ) for config_path in config_paths
    ]

    os.chdir(REPOSITORY_ROOT)
    benchmark_start = time.perf_counter()

    for index, (config_path, config) in enumerate(configs, start=1):
        print(
            f"\n[benchmark {index}/{len(configs)}] {config_path}",
            flush=True,
        )
        run_start = time.perf_counter()
        ExperimentRunner(config, source_config_path=config_path).run()
        elapsed = time.perf_counter() - run_start
        print(f"[benchmark] Finished in {elapsed:.1f} seconds.", flush=True)

    elapsed = time.perf_counter() - benchmark_start
    print(
        f"\n[benchmark] Completed {len(configs)} configurations "
        f"in {elapsed:.1f} seconds.",
        flush=True,
    )


if __name__ == "__main__":
    main()
