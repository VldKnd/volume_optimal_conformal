from __future__ import annotations

import json
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from configs.datasets.real import QM9DatasetConfig
from data.datasets.real.base import BaseRealDataset


class QM9Dataset(BaseRealDataset):
    """QM9 targets with cached non-neural MBTR molecular features.

    The configured NPZ stores the sparse descriptors before SVD so it remains
    reusable across split seeds. ``load_data`` fits the 95-component projection
    on exactly the training rows selected by ``BaseRealDataset.prepare``.
    """

    dataset_name = "QM9"
    target_names = ("mu", "alpha", "homo", "lumo")
    target_indices = (0, 1, 2, 3)
    target_units = ("Debye", "Bohr^3", "eV", "eV")

    species = ("H", "C", "N", "O", "F")
    atomic_numbers = (1, 6, 7, 8, 9)
    element_feature_names = tuple(f"count_{element}" for element in species)
    svd_feature_names = tuple(f"mbtr_svd_{index:03d}" for index in range(95))
    final_feature_names = element_feature_names + svd_feature_names

    cache_schema_version = 1
    svd_components = 95

    two_body_parameters = {
        "geometry": {
            "function": "inverse_distance"
        },
        "grid": {
            "min": 0.0,
            "max": 1.6,
            "n": 40,
            "sigma": 0.05,
        },
        "weighting": {
            "function": "exp",
            "scale": 0.5,
            "threshold": 1e-3,
        },
    }
    three_body_parameters = {
        "geometry": {
            "function": "cosine"
        },
        "grid": {
            "min": -1.0,
            "max": 1.0,
            "n": 30,
            "sigma": 0.05,
        },
        "weighting": {
            "function": "exp",
            "scale": 0.5,
            "threshold": 1e-3,
        },
    }

    def __init__(self, config: QM9DatasetConfig):
        super().__init__(config)
        self.config = config
        self.raw_root = self._resolve_path(config.raw_root)
        self.svd: TruncatedSVD | None = None
        self._cache_ready = False

    def prepare(self) -> None:
        self._ensure_cache()
        super().prepare()

    def load_data(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        self._ensure_cache()
        element_counts, mbtr, y = self._load_cache()

        n_samples = y.shape[0]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        permutation = torch.randperm(
            n_samples,
            generator=generator,
        ).numpy()
        n_train = int(self.config.train_fraction * n_samples)
        train_indices = permutation[:n_train]

        if n_train <= self.svd_components:
            raise ValueError(
                "QM9 needs more than 95 training molecules to fit the requested "
                f"TruncatedSVD, but the configured split contains {n_train}."
            )
        if mbtr.shape[1] < self.svd_components:
            raise ValueError(
                "QM9 raw MBTR cache has only "
                f"{mbtr.shape[1]} columns; at least {self.svd_components} are "
                "required. Rebuild the cache with the configured descriptors."
            )

        self.svd = TruncatedSVD(
            n_components=self.svd_components,
            random_state=self.config.seed,
        )
        try:
            self.svd.fit(mbtr[train_indices])
            mbtr_components = self.svd.transform(mbtr)
        except Exception as error:
            raise RuntimeError(
                "Failed to fit or apply QM9 TruncatedSVD using only the "
                f"{n_train} configured training rows."
            ) from error

        x = np.concatenate(
            [
                element_counts,
                np.asarray(mbtr_components, dtype=np.float32),
            ],
            axis=1,
            dtype=np.float32,
        )
        if x.shape != (n_samples, self.x_dim):
            raise ValueError(
                f"QM9 feature extraction produced shape {x.shape}, expected "
                f"({n_samples}, {self.x_dim})."
            )
        if not np.isfinite(x).all():
            raise ValueError(
                "QM9 count or SVD features contain non-finite values; the raw "
                "cache may be corrupted."
            )

        return x, y, self.final_feature_names

    def _resolve_path(self, path: Path | None) -> Path:
        if path is None:
            raise ValueError("QM9 raw_root could not be resolved from the config.")
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[4] / path

    def _ensure_cache(self) -> None:
        if self._cache_ready:
            return
        if self.config.force_rebuild_features or not self.file_path.is_file():
            self._build_cache()
        self._cache_ready = True

    def _build_cache(self) -> None:
        Atoms, MBTR, PyGQM9 = self._load_extraction_dependencies()
        self.raw_root.mkdir(parents=True, exist_ok=True)
        try:
            qm9 = PyGQM9(root=str(self.raw_root))
        except Exception as error:
            raise RuntimeError(
                "Failed to download or preprocess the curated PyG QM9 dataset "
                f"under '{self.raw_root}'. Check network access, available disk "
                "space, and the torch-geometric installation."
            ) from error

        n_samples = len(qm9)
        if n_samples <= 0:
            raise ValueError("The curated PyG QM9 dataset contains no molecules.")

        try:
            two_body, three_body = self._make_mbtr_descriptors(MBTR)
        except Exception as error:
            raise RuntimeError(
                "Failed to initialize the fixed two-body and three-body QM9 "
                "MBTR descriptors. Check the installed dscribe version."
            ) from error
        element_counts = np.empty(
            (n_samples, len(self.species)),
            dtype=np.float32,
        )
        y = np.empty((n_samples, len(self.target_indices)), dtype=np.float32)
        mbtr_batches: list[sparse.csr_matrix] = []
        source_target_dimension: int | None = None

        batch_size = self.config.extraction_batch_size
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            atoms_batch = []

            for molecule_index in range(batch_start, batch_end):
                try:
                    molecule = qm9[molecule_index]
                    atomic_numbers = molecule.z.detach().cpu().numpy()
                    positions = molecule.pos.detach().cpu().numpy()
                    targets = molecule.y.detach().cpu().view(-1)
                except Exception as error:
                    raise RuntimeError(
                        "Failed to read QM9 molecule "
                        f"{molecule_index} from the curated PyG dataset."
                    ) from error

                if targets.numel() <= max(self.target_indices):
                    raise ValueError(
                        f"QM9 molecule {molecule_index} exposes "
                        f"{targets.numel()} targets; indices "
                        f"{self.target_indices} are required."
                    )
                if source_target_dimension is None:
                    source_target_dimension = targets.numel()
                elif targets.numel() != source_target_dimension:
                    raise ValueError(
                        f"QM9 molecule {molecule_index} exposes "
                        f"{targets.numel()} targets, while earlier molecules "
                        f"exposed {source_target_dimension}."
                    )
                selected_targets = targets[list(self.target_indices)].numpy()
                if not np.isfinite(selected_targets).all():
                    raise ValueError(
                        "QM9 selected targets contain non-finite values at "
                        f"molecule {molecule_index}."
                    )

                unknown_atomic_numbers = set(np.unique(atomic_numbers)
                                             ) - set(self.atomic_numbers)
                if unknown_atomic_numbers:
                    raise ValueError(
                        f"QM9 molecule {molecule_index} contains unsupported "
                        "atomic numbers "
                        f"{sorted(unknown_atomic_numbers)}; expected only "
                        f"{self.atomic_numbers}."
                    )

                y[molecule_index] = selected_targets
                element_counts[molecule_index] = [
                    np.count_nonzero(atomic_numbers == atomic_number)
                    for atomic_number in self.atomic_numbers
                ]
                try:
                    atoms_batch.append(
                        Atoms(
                            numbers=atomic_numbers,
                            positions=positions,
                            pbc=False,
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Failed to convert QM9 molecule "
                        f"{molecule_index} to an ASE Atoms object."
                    ) from error

            try:
                two_body_batch = self._descriptor_to_csr(
                    two_body.create(
                        atoms_batch,
                        n_jobs=self.config.extraction_n_jobs,
                    ),
                    expected_rows=batch_end - batch_start,
                )
                three_body_batch = self._descriptor_to_csr(
                    three_body.create(
                        atoms_batch,
                        n_jobs=self.config.extraction_n_jobs,
                    ),
                    expected_rows=batch_end - batch_start,
                )
                mbtr_batch = sparse.hstack(
                    [two_body_batch, three_body_batch],
                    format="csr",
                    dtype=np.float32,
                )
            except Exception as error:
                raise RuntimeError(
                    "MBTR extraction failed for QM9 molecule batch "
                    f"[{batch_start}, {batch_end})."
                ) from error

            expected_rows = batch_end - batch_start
            if mbtr_batch.shape[0] != expected_rows:
                raise ValueError(
                    "MBTR returned "
                    f"{mbtr_batch.shape[0]} rows for QM9 batch "
                    f"[{batch_start}, {batch_end}), expected {expected_rows}."
                )
            mbtr_batches.append(mbtr_batch)

        mbtr = sparse.vstack(
            mbtr_batches,
            format="csr",
            dtype=np.float32,
        )
        if not np.isfinite(element_counts).all() or not np.isfinite(mbtr.data).all():
            raise ValueError(
                "QM9 MBTR extraction produced non-finite count or descriptor "
                "values."
            )

        metadata = self._cache_metadata(
            sample_count=n_samples,
            mbtr_shape=mbtr.shape,
            source_target_dimension=(
                source_target_dimension
                if source_target_dimension is not None else len(self.target_indices)
            ),
        )
        self._save_cache(
            element_counts=element_counts,
            mbtr=mbtr,
            y=y,
            metadata=metadata,
        )

    @staticmethod
    def _load_extraction_dependencies() -> tuple[Any, Any, Any]:
        try:
            from ase import Atoms
            from dscribe.descriptors import MBTR
            from torch_geometric.datasets import QM9 as PyGQM9
        except (ImportError, OSError) as error:
            raise ImportError(
                "Building the QM9 MBTR cache requires torch-geometric, dscribe, "
                "and ase. Install the project dependencies with `uv sync`. An "
                "existing QM9 cache can be loaded without these extraction "
                "dependencies."
            ) from error
        return Atoms, MBTR, PyGQM9

    def _make_mbtr_descriptors(self, MBTR: Any) -> tuple[Any, Any]:
        common = {
            "species": list(self.species),
            "periodic": False,
            "normalization": "none",
            "sparse": True,
            "dtype": "float32",
        }
        two_body = MBTR(**common, **self.two_body_parameters)
        three_body = MBTR(**common, **self.three_body_parameters)
        return two_body, three_body

    @staticmethod
    def _descriptor_to_csr(
        descriptor: Any,
        expected_rows: int,
    ) -> sparse.csr_matrix:
        if sparse.issparse(descriptor):
            matrix = descriptor.tocsr()
        elif hasattr(descriptor, "to_scipy_sparse"):
            matrix = descriptor.to_scipy_sparse().tocsr()
        elif hasattr(descriptor, "tocsr"):
            matrix = descriptor.tocsr()
        else:
            matrix = sparse.csr_matrix(descriptor)
        if matrix.ndim == 1:
            if expected_rows != 1:
                raise ValueError(
                    f"MBTR returned a one-dimensional descriptor for "
                    f"{expected_rows} molecules."
                )
            matrix = matrix.reshape(1, -1)
        return sparse.csr_matrix(matrix, dtype=np.float32, copy=False)

    def _cache_metadata(
        self,
        sample_count: int,
        mbtr_shape: tuple[int, int],
        source_target_dimension: int = 19,
    ) -> dict[str, Any]:
        package_versions = {}
        for package_name in ("ase", "dscribe", "torch-geometric"):
            try:
                package_versions[package_name] = importlib_metadata.version(
                    package_name
                )
            except importlib_metadata.PackageNotFoundError:
                package_versions[package_name] = None

        return {
            "schema_version": self.cache_schema_version,
            "cache_kind": "pre_svd_sparse_mbtr",
            "dataset_name": self.dataset_name,
            "sample_count": sample_count,
            "expected_sample_count_approximately": 130_831,
            "sample_order": "torch_geometric.datasets.QM9 order",
            "target_names": list(self.target_names),
            "target_indices": list(self.target_indices),
            "target_units": list(self.target_units),
            "source_target_dimension": source_target_dimension,
            "species": list(self.species),
            "element_count_features": list(self.element_feature_names),
            "raw_mbtr_shape": list(mbtr_shape),
            "feature_extraction": self._feature_extraction_metadata(),
            "package_versions": package_versions,
        }

    def _feature_extraction_metadata(self) -> dict[str, Any]:
        return {
            "two_body_mbtr": self.two_body_parameters,
            "three_body_mbtr": self.three_body_parameters,
            "periodic": False,
            "normalization": "none",
            "sparse": True,
            "dtype": "float32",
            "svd_components": self.svd_components,
            "svd_fit_scope": "configured training split only",
            "standardized_during_extraction": False,
        }

    def _save_cache(
        self,
        element_counts: np.ndarray,
        mbtr: sparse.csr_matrix,
        y: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix=f".{self.file_path.stem}_",
                dir=self.file_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.savez_compressed(
                    temporary_file,
                    element_counts=np.asarray(element_counts, dtype=np.float32),
                    mbtr_data=np.asarray(mbtr.data, dtype=np.float32),
                    mbtr_indices=mbtr.indices,
                    mbtr_indptr=mbtr.indptr,
                    mbtr_shape=np.asarray(mbtr.shape, dtype=np.int64),
                    y=np.asarray(y, dtype=np.float32),
                    feature_names=np.asarray(self.final_feature_names),
                    target_names=np.asarray(self.target_names),
                    target_indices=np.asarray(self.target_indices, dtype=np.int64),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            temporary_path.replace(self.file_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _load_cache(self) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
        try:
            with np.load(self.file_path, allow_pickle=False) as cache:
                required_keys = {
                    "element_counts",
                    "mbtr_data",
                    "mbtr_indices",
                    "mbtr_indptr",
                    "mbtr_shape",
                    "y",
                    "feature_names",
                    "target_names",
                    "target_indices",
                    "metadata_json",
                }
                missing_keys = required_keys - set(cache.files)
                if missing_keys:
                    raise ValueError(f"missing keys {sorted(missing_keys)}")

                element_counts = np.asarray(
                    cache["element_counts"],
                    dtype=np.float32,
                )
                y = np.asarray(cache["y"], dtype=np.float32)
                feature_names = tuple(cache["feature_names"].tolist())
                target_names = tuple(cache["target_names"].tolist())
                target_indices = tuple(cache["target_indices"].tolist())
                metadata = json.loads(str(cache["metadata_json"].item()))
                mbtr_shape = tuple(int(dimension) for dimension in cache["mbtr_shape"])
                mbtr = sparse.csr_matrix(
                    (
                        np.asarray(cache["mbtr_data"], dtype=np.float32),
                        cache["mbtr_indices"],
                        cache["mbtr_indptr"],
                    ),
                    shape=mbtr_shape,
                    dtype=np.float32,
                )
        except Exception as error:
            if isinstance(error,
                          ValueError) and str(error).startswith("Invalid QM9 cache"):
                raise
            raise ValueError(
                f"Invalid QM9 cache at '{self.file_path}': {error}. Set "
                "force_rebuild_features=true to rebuild it."
            ) from error

        self._validate_cache(
            element_counts=element_counts,
            mbtr=mbtr,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            target_indices=target_indices,
            metadata=metadata,
        )
        return element_counts, mbtr, y

    def _validate_cache(
        self,
        element_counts: np.ndarray,
        mbtr: sparse.csr_matrix,
        y: np.ndarray,
        feature_names: tuple[str, ...],
        target_names: tuple[str, ...],
        target_indices: tuple[int, ...],
        metadata: dict[str, Any],
    ) -> None:
        n_samples = y.shape[0] if y.ndim == 2 else -1
        problems = []
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid QM9 cache at '{self.file_path}': metadata_json must "
                "contain a JSON object. Set force_rebuild_features=true to "
                "rebuild it."
            )
        if metadata.get("schema_version") != self.cache_schema_version:
            problems.append("unsupported schema version")
        if metadata.get("cache_kind") != "pre_svd_sparse_mbtr":
            problems.append("unexpected cache kind")
        if metadata.get("dataset_name") != self.dataset_name:
            problems.append("dataset name mismatch")
        if metadata.get("sample_count") != n_samples:
            problems.append("sample count metadata mismatch")
        if tuple(metadata.get("target_names", ())) != self.target_names:
            problems.append("target-name metadata mismatch")
        if tuple(metadata.get("target_indices", ())) != self.target_indices:
            problems.append("target-index metadata mismatch")
        if tuple(metadata.get("target_units", ())) != self.target_units:
            problems.append("target-unit metadata mismatch")
        source_target_dimension = metadata.get("source_target_dimension")
        if (
            not isinstance(source_target_dimension, int)
            or source_target_dimension <= max(self.target_indices)
        ):
            problems.append("invalid source target dimension")
        if tuple(metadata.get("species", ())) != self.species:
            problems.append("species metadata mismatch")
        if tuple(
            metadata.get("element_count_features", ())
        ) != self.element_feature_names:
            problems.append("element-count metadata mismatch")
        if metadata.get("feature_extraction") != self._feature_extraction_metadata():
            problems.append("feature-extraction metadata mismatch")
        if tuple(metadata.get("raw_mbtr_shape", ())) != mbtr.shape:
            problems.append("raw MBTR shape metadata mismatch")
        if feature_names != self.final_feature_names:
            problems.append("feature names mismatch")
        if target_names != self.target_names:
            problems.append("target names mismatch")
        if target_indices != self.target_indices:
            problems.append("target indices mismatch")
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            problems.append(f"target shape is {y.shape}")
        if element_counts.shape != (n_samples, len(self.species)):
            problems.append(f"element-count shape is {element_counts.shape}")
        if mbtr.shape[0] != n_samples:
            problems.append(f"MBTR row count is {mbtr.shape[0]}")
        if not np.isfinite(y).all():
            problems.append("selected targets are non-finite")
        if not np.isfinite(element_counts).all():
            problems.append("element counts are non-finite")
        if not np.isfinite(mbtr.data).all():
            problems.append("MBTR values are non-finite")

        if problems:
            raise ValueError(
                f"Invalid QM9 cache at '{self.file_path}': "
                f"{'; '.join(problems)}. Set force_rebuild_features=true to "
                "rebuild it."
            )
