from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from denoistpy import denoist, summarize_denoist_result


REFERENCE_DIR = Path("tests_py/parity/reference")


def _read_matrix_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _require_reference():
    required = [
        "raw_counts.csv",
        "coords.csv",
        "transcripts.csv",
        "adjusted_counts_fast.csv",
        "memberships_fast.csv",
        "params_fast.csv",
    ]
    missing = [name for name in required if not (REFERENCE_DIR / name).exists()]
    if missing:
        pytest.skip(
            "R parity reference files are missing. Generate them with "
            "`conda run -n denoistpy_r Rscript tests_py/parity/generate_r_reference.R`."
        )


def test_python_roughly_matches_r_reference_totals():
    _require_reference()
    raw = _read_matrix_csv(REFERENCE_DIR / "raw_counts.csv")
    coords = pd.read_csv(REFERENCE_DIR / "coords.csv", index_col=0)
    tx = pd.read_csv(REFERENCE_DIR / "transcripts.csv")
    r_adjusted = _read_matrix_csv(REFERENCE_DIR / "adjusted_counts_fast.csv")
    gene_names = raw.index.to_numpy()
    cell_names = raw.columns.to_numpy()

    raw_counts = sparse.csr_matrix(raw.to_numpy(dtype=np.float32))
    r_adjusted_counts = sparse.csr_matrix(r_adjusted.to_numpy(dtype=np.float32))

    py_result = denoist(
        raw_counts,
        tx,
        coords.to_numpy(dtype=float),
        gene_names=gene_names,
        cell_names=cell_names,
        distance=50,
        nbins=200,
        posterior_cutoff=0.6,
        n_inits=np.array([0.1, 0.2, 0.3]),
        backend="numpy",
        include_self_twice=True,
        store_memberships=True,
        store_posterior=False,
    )
    py_report = summarize_denoist_result(py_result, raw_counts)
    r_removed_total = float(raw_counts.sum() - r_adjusted_counts.sum())
    py_removed_total = float(
        py_report["summary"].set_index("metric").loc["removed_counts_total", "value"]
    )

    assert raw_counts.shape == r_adjusted_counts.shape
    if r_removed_total == 0:
        assert py_removed_total == 0
    else:
        rel_error = abs(py_removed_total - r_removed_total) / r_removed_total
        assert rel_error < 0.25


def test_r_reference_has_expected_membership_shape():
    _require_reference()
    raw = _read_matrix_csv(REFERENCE_DIR / "raw_counts.csv")
    memberships = _read_matrix_csv(REFERENCE_DIR / "memberships_fast.csv")
    params = pd.read_csv(REFERENCE_DIR / "params_fast.csv")

    assert raw.shape == memberships.shape
    assert len(params) == raw.shape[1]
