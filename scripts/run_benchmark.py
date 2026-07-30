from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from experimentation import ExperimentRunner, load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark configurations sequentially."
    )
    parser.add_argument(
        "config_directory",
        type=Path,
        help="Directory containing YAML configurations.",
    )
    args = parser.parse_args()

    config_directory = args.config_directory
    if not config_directory.is_absolute():
        config_directory = REPOSITORY_ROOT / config_directory

    config_paths = sorted(config_directory.rglob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(
            f"No YAML configurations found in {config_directory}."
        )

    configs = [
        (config_path, load_experiment_config(config_path))
        for config_path in config_paths
    ]

    os.chdir(REPOSITORY_ROOT)
    benchmark_start = time.perf_counter()

    for index, (config_path, config) in enumerate(configs, start=1):
        print(
            f"\n[benchmark {index}/{len(configs)}] {config_path}",
            flush=True,
        )
        run_start = time.perf_counter()
        ExperimentRunner(config).run()
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
