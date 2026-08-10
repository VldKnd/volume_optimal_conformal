from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from pydantic import TypeAdapter, ValidationError
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from configs.datasets.real import QM9DatasetConfig
from data.datasets.real import QM9Dataset
from experimentation.config import DatasetConfig


class _FakeAtoms:

    def __init__(self, numbers, positions, pbc):
        self.numbers = np.asarray(numbers)
        self.positions = np.asarray(positions)
        self.pbc = pbc


class _FakeMolecule:

    def __init__(self, index: int, non_finite_target: bool = False):
        self.z = torch.tensor([1, 1, 6, 8, 9], dtype=torch.long)
        self.pos = torch.tensor(
            [[float(index), float(atom), 0.0] for atom in range(5)],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[index + 0.1, index + 0.2, index + 0.3, index + 0.4]],
            dtype=torch.float32,
        )
        if non_finite_target:
            targets[0, 2] = torch.nan
        self.y = targets


class _FakeQM9:
    sample_count = 5
    non_finite_index: int | None = None
    roots: list[str] = []

    def __init__(self, root: str):
        self.roots.append(root)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index: int):
        return _FakeMolecule(
            index,
            non_finite_target=index == self.non_finite_index,
        )


class _FakeMBTR:
    create_calls: list[tuple[str, int, int]] = []

    def __init__(self, **kwargs):
        self.kind = kwargs["geometry"]["function"]
        self.n_features = 4 if self.kind == "inverse_distance" else 6

    def create(self, systems, n_jobs):
        self.create_calls.append((self.kind, len(systems), n_jobs))
        rows = []
        for atoms in systems:
            base = float(atoms.positions[0, 0] + 1.0)
            rows.append(base * np.arange(1, self.n_features + 1))
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


class QM9DatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        _FakeQM9.roots.clear()
        _FakeQM9.non_finite_index = None
        _FakeMBTR.create_calls.clear()

    def test_config_defaults_and_registration(self) -> None:
        config = QM9DatasetConfig()

        self.assertEqual(config.file_path, Path("data/qm9/qm9_mbtr_100.npz"))
        self.assertEqual(config.raw_root, Path("data/qm9/pyg"))
        self.assertEqual(config.x_dim, 100)
        self.assertEqual(config.y_dim, 4)
        self.assertEqual(config.extraction_batch_size, 1024)
        self.assertEqual(config.extraction_n_jobs, -1)
        self.assertFalse(config.force_rebuild_features)
        self.assertEqual(QM9Dataset.target_names, ("mu", "alpha", "homo", "lumo"))

        parsed = TypeAdapter(DatasetConfig).validate_python({"type": "qm9"})
        self.assertIsInstance(parsed, QM9DatasetConfig)

        with self.assertRaises(ValidationError):
            QM9DatasetConfig(x_dim=99)
        with self.assertRaises(ValidationError):
            QM9DatasetConfig(y_dim=3)
        with self.assertRaisesRegex(ValidationError, "extraction_n_jobs"):
            QM9DatasetConfig(extraction_n_jobs=0)

    def test_builds_pre_svd_sparse_cache_in_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = QM9DatasetConfig(
                file_path=root / "qm9_mbtr_100.npz",
                raw_root=root / "pyg",
                extraction_batch_size=2,
                extraction_n_jobs=-1,
            )
            dataset = QM9Dataset(config)

            with mock.patch.object(
                QM9Dataset,
                "_load_extraction_dependencies",
                return_value=(_FakeAtoms, _FakeMBTR, _FakeQM9),
            ):
                dataset._build_cache()

            with np.load(config.file_path, allow_pickle=False) as cache:
                self.assertEqual(cache["element_counts"].shape, (5, 5))
                self.assertEqual(cache["y"].shape, (5, 4))
                self.assertEqual(
                    tuple(cache["feature_names"]), QM9Dataset.final_feature_names
                )
                self.assertEqual(tuple(cache["target_indices"]), (0, 1, 2, 3))
                self.assertEqual(tuple(cache["mbtr_shape"]), (5, 10))
                metadata = json.loads(str(cache["metadata_json"].item()))

            self.assertEqual(metadata["cache_kind"], "pre_svd_sparse_mbtr")
            self.assertEqual(metadata["sample_count"], 5)
            self.assertEqual(metadata["species"], ["H", "C", "N", "O", "F"])
            self.assertEqual(
                _FakeMBTR.create_calls,
                [
                    ("inverse_distance", 2, -1),
                    ("cosine", 2, -1),
                    ("inverse_distance", 2, -1),
                    ("cosine", 2, -1),
                    ("inverse_distance", 1, -1),
                    ("cosine", 1, -1),
                ],
            )
            self.assertEqual(_FakeQM9.roots, [str(config.raw_root)])

            counts, mbtr, targets = dataset._load_cache()
            np.testing.assert_array_equal(counts[0], [2, 1, 0, 1, 1])
            self.assertTrue(sparse.isspmatrix_csr(mbtr))
            np.testing.assert_allclose(targets[4], [4.1, 4.2, 4.3, 4.4])

    def test_cached_mbtr_is_reused_and_svd_fits_only_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "qm9_mbtr_100.npz"
            config = QM9DatasetConfig(file_path=path, seed=37)
            dataset = QM9Dataset(config)
            rng = np.random.default_rng(731)
            n_samples = 170
            element_counts = rng.integers(
                0,
                8,
                size=(n_samples, 5),
            ).astype(np.float32)
            raw_mbtr = sparse.csr_matrix(
                rng.normal(size=(n_samples, 100)).astype(np.float32)
            )
            targets = rng.normal(size=(n_samples, 4)).astype(np.float32)
            dataset._save_cache(
                element_counts=element_counts,
                mbtr=raw_mbtr,
                y=targets,
                metadata=dataset._cache_metadata(
                    sample_count=n_samples,
                    mbtr_shape=raw_mbtr.shape,
                ),
            )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)
            permutation = torch.randperm(
                n_samples,
                generator=generator,
            ).numpy()
            n_train = int(config.train_fraction * n_samples)
            expected_training_mbtr = raw_mbtr[permutation[:n_train]]

            original_fit = TruncatedSVD.fit
            fitted_rows = {}

            def recording_fit(svd, matrix, y=None):
                fitted_rows["matrix"] = matrix.copy()
                return original_fit(svd, matrix, y)

            with (
                mock.patch.object(
                    QM9Dataset,
                    "_load_extraction_dependencies",
                    side_effect=AssertionError(
                        "cached loading must not import extraction dependencies"
                    ),
                ),
                mock.patch.object(TruncatedSVD, "fit", new=recording_fit),
            ):
                x, y, feature_names = dataset.load_data()

            difference = fitted_rows["matrix"] - expected_training_mbtr
            self.assertEqual(difference.nnz, 0)
            self.assertEqual(x.shape, (n_samples, 100))
            np.testing.assert_array_equal(x[:, :5], element_counts)
            np.testing.assert_array_equal(y, targets)
            self.assertEqual(feature_names, QM9Dataset.final_feature_names)
            self.assertIsNotNone(dataset.svd)

    def test_non_finite_selected_target_identifies_molecule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = QM9Dataset(
                QM9DatasetConfig(
                    file_path=root / "cache.npz",
                    raw_root=root / "pyg",
                )
            )
            _FakeQM9.non_finite_index = 3

            with mock.patch.object(
                QM9Dataset,
                "_load_extraction_dependencies",
                return_value=(_FakeAtoms, _FakeMBTR, _FakeQM9),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "non-finite values at molecule 3",
                ):
                    dataset._build_cache()

    def test_real_mbtr_descriptors_produce_expected_sparse_blocks(self) -> None:
        from ase import Atoms
        from dscribe.descriptors import MBTR

        dataset = QM9Dataset(QM9DatasetConfig())
        two_body, three_body = dataset._make_mbtr_descriptors(MBTR)
        atoms = Atoms(
            numbers=[6, 1, 1, 1, 1],
            positions=[
                [0.0, 0.0, 0.0],
                [0.6, 0.6, 0.6],
                [-0.6, -0.6, 0.6],
                [-0.6, 0.6, -0.6],
                [0.6, -0.6, -0.6],
            ],
            pbc=False,
        )

        two_body_matrix = dataset._descriptor_to_csr(
            two_body.create([atoms], n_jobs=1),
            expected_rows=1,
        )
        three_body_matrix = dataset._descriptor_to_csr(
            three_body.create([atoms], n_jobs=1),
            expected_rows=1,
        )

        self.assertEqual(two_body_matrix.shape, (1, 600))
        self.assertEqual(three_body_matrix.shape, (1, 2250))
        self.assertEqual(two_body_matrix.dtype, np.float32)
        self.assertEqual(three_body_matrix.dtype, np.float32)
        self.assertTrue(np.isfinite(two_body_matrix.data).all())
        self.assertTrue(np.isfinite(three_body_matrix.data).all())


if __name__ == "__main__":
    unittest.main()
