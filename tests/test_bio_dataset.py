from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from configs.datasets.real import BioDatasetConfig
from data.datasets.real import BioDataset


class BioDatasetTest(unittest.TestCase):

    def test_loads_casp_csv_and_selects_f7_and_f9_as_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "CASP.csv"
            path.write_text(
                "RMSD,F1,F2,F3,F4,F5,F6,F7,F8,F9\n"
                "1,2,3,4,5,6,7,8,9,10\n"
                "11,12,13,14,15,16,17,18,19,20\n",
                encoding="utf-8",
            )
            dataset = BioDataset(BioDatasetConfig(file_path=path))

            x, y, feature_names = dataset.load_data()

        self.assertEqual(
            feature_names,
            ("RMSD", "F1", "F2", "F3", "F4", "F5", "F6", "F8"),
        )
        np.testing.assert_array_equal(
            x,
            np.asarray(
                [
                    [1, 2, 3, 4, 5, 6, 7, 9],
                    [11, 12, 13, 14, 15, 16, 17, 19],
                ]
            ),
        )
        np.testing.assert_array_equal(y, np.asarray([[8, 10], [18, 20]]))


if __name__ == "__main__":
    unittest.main()
