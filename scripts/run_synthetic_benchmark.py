from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from experimentation import (  # noqa: E402
    SyntheticExperimentRunner,
    load_experiment_config,
)
from scripts.run_benchmark import (  # noqa: E402
    _apply_wandb_overrides,
    _discover_config_paths,
)


def run_synthetic_benchmark(
    config,
    source_config_path: str | Path | None = None,
) -> SyntheticExperimentRunner:
    """Train and evaluate one supported synthetic experiment configuration."""
    return SyntheticExperimentRunner(
        config=config,
        source_config_path=source_config_path,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run banana, sinusoidal, or star-shaped synthetic benchmarks "
            "with an analytic HDR-volume comparison."
        )
    )
    parser.add_argument(
        "config_directory",
        type=Path,
        help="One YAML configuration or a directory searched recursively.",
    )
    parser.add_argument(
        "--priority-method",
        action="append",
        default=[],
        metavar="METHOD",
        help=(
            "Run configurations whose parent directory has this name before "
            "other configurations; repeat to specify priority order."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default=None,
        help="Override W&B mode for every selected configuration.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", nargs="+", default=None, metavar="TAG")
    args = parser.parse_args()

    config_directory = args.config_directory
    if not config_directory.is_absolute():
        config_directory = REPOSITORY_ROOT / config_directory

    config_paths = _discover_config_paths(
        config_directory,
        priority_method_names=tuple(args.priority_method),
    )
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
            f"\n[synthetic benchmark {index}/{len(configs)}] {config_path}",
            flush=True,
        )
        run_start = time.perf_counter()
        run_synthetic_benchmark(config, source_config_path=config_path)
        elapsed = time.perf_counter() - run_start
        print(
            f"[synthetic benchmark] Finished in {elapsed:.1f} seconds.",
            flush=True,
        )

    elapsed = time.perf_counter() - benchmark_start
    print(
        f"\n[synthetic benchmark] Completed {len(configs)} configurations "
        f"in {elapsed:.1f} seconds.",
        flush=True,
    )


if __name__ == "__main__":
    main()
