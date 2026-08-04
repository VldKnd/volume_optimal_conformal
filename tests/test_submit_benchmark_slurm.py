from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = REPOSITORY_ROOT / "scripts/submit_benchmark_slurm.sh"
WORKER = REPOSITORY_ROOT / "scripts/run_experiment_slurm.sh"


class SubmitBenchmarkSlurmTest(unittest.TestCase):

    def test_dry_run_submits_one_gpu_job_per_configuration(self):
        config_directory = (
            REPOSITORY_ROOT
            / "benchmark/configurations/scm20d/continuing_only_rearranged_realnvp"
            / "transport_realnvp_rearranged_l2"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = os.environ.copy()
            environment.update(
                MVCP_SLURM_ACCOUNT="nils",
                MVCP_SLURM_LOG_DIRECTORY=temporary_directory,
            )
            result = subprocess.run(
                ["bash", str(SUBMITTER), "--dry-run", str(config_directory)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

        commands = result.stdout.splitlines()
        self.assertEqual(len(commands), 10)
        self.assertTrue(all("--gres=gpu:1" in command for command in commands))
        self.assertTrue(all("run_experiment_slurm.sh" in command for command in commands))
        self.assertNotIn("--gres=gpu:2", result.stdout)
        self.assertNotIn(" srun ", result.stdout)
        self.assertNotIn("--dependency", commands[0])
        self.assertNotIn("--dependency", commands[1])
        self.assertIn("--dependency=afterany:DRY_RUN_0", commands[2])
        self.assertIn("--dependency=afterany:DRY_RUN_1", commands[3])

    def test_rejects_more_than_cluster_submission_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_directory = Path(temporary_directory) / "configs"
            config_directory.mkdir()
            for index in range(21):
                (config_directory / f"seed_{index:02d}.yaml").write_text(
                    "name: placeholder\n",
                    encoding="utf-8",
                )

            result = subprocess.run(
                ["bash", str(SUBMITTER), "--dry-run", str(config_directory)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("at most 20", result.stderr)

    def test_worker_runs_exactly_one_configuration(self):
        config_path = (
            REPOSITORY_ROOT
            / "benchmark/configurations/scm20d/continuing_only_rearranged_realnvp"
            / "transport_realnvp_rearranged_l2/seed_00.yaml"
        )
        environment = os.environ.copy()
        environment.update(MVCP_PYTHON_EXECUTABLE="/bin/echo")
        result = subprocess.run(
            ["bash", str(WORKER), str(config_path)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("scripts/run_benchmark.py", result.stdout)
        self.assertIn(str(config_path), result.stdout)
        self.assertNotIn("--wandb-mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
