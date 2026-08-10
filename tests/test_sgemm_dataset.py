import tempfile
import unittest
from pathlib import Path

import numpy as np

from configs.datasets.real import SGEMMDatasetConfig
from data.datasets.real import SGEMMDataset


class SGEMMDatasetTest(unittest.TestCase):

    column_names = tuple(f"feature_{index}" for index in range(14)) + (
        "Run1 (ms)",
        "Run2 (ms)",
        "Run3 (ms)",
        "Run4 (ms)",
    )

    def _write_dataset(self, path: Path, targets: np.ndarray) -> None:
        features = np.arange(targets.shape[0] * 14,
                             dtype=np.float64).reshape(targets.shape[0], 14)
        np.savetxt(
            path,
            np.column_stack([features, targets]),
            delimiter=",",
            header=",".join(self.column_names),
            comments="",
        )

    def test_log_transforms_all_runtime_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sgemm.csv"
            targets = np.asarray([
                [1.0, np.e, np.e**2, 10.0],
                [2.0, 4.0, 8.0, 16.0],
            ])
            self._write_dataset(path, targets)

            x, y, feature_names = SGEMMDataset(SGEMMDatasetConfig(file_path=path)
                                               ).load_data()

            np.testing.assert_allclose(y, np.log(targets))
            self.assertEqual(x.shape, (2, 14))
            self.assertEqual(feature_names, self.column_names[:14])

    def test_rejects_non_positive_runtime_before_log_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sgemm.csv"
            targets = np.ones((1, 4))
            targets[0, 2] = 0.0
            self._write_dataset(path, targets)

            dataset = SGEMMDataset(SGEMMDatasetConfig(file_path=path))
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                dataset.load_data()


if __name__ == "__main__":
    unittest.main()
