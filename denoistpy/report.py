from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import sparse

from .io import DenoistInput
from .types import DenoistResult


def _coerce_raw_counts(raw_counts: sparse.spmatrix | np.ndarray | DenoistInput) -> sparse.csr_matrix:
    if isinstance(raw_counts, DenoistInput):
        raw_counts = raw_counts.counts
    return sparse.csr_matrix(raw_counts)


def _names(names: np.ndarray | None, prefix: str, n: int) -> np.ndarray:
    if names is None:
        return np.asarray([f"{prefix}{idx}" for idx in range(n)], dtype=object)
    return np.asarray([str(x) for x in names], dtype=object)


def summarize_denoist_result(
    result: DenoistResult,
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    *,
    top_n: int = 25,
) -> dict[str, pd.DataFrame]:
    """Summarize removed count mass from a DenoIST result.

    The report treats count differences as removed transcript-count mass. The
    current DenoIST model does not classify individual transcript records.
    """

    raw = _coerce_raw_counts(raw_counts)
    adjusted = sparse.csr_matrix(result.adjusted_counts)
    if raw.shape != adjusted.shape:
        raise ValueError(f"raw_counts shape {raw.shape} does not match adjusted_counts shape {adjusted.shape}.")

    removed = (raw - adjusted).maximum(0).tocsr()
    raw_gene = np.asarray(raw.sum(axis=1)).ravel()
    adj_gene = np.asarray(adjusted.sum(axis=1)).ravel()
    rem_gene = np.asarray(removed.sum(axis=1)).ravel()
    raw_cell = np.asarray(raw.sum(axis=0)).ravel()
    adj_cell = np.asarray(adjusted.sum(axis=0)).ravel()
    rem_cell = np.asarray(removed.sum(axis=0)).ravel()
    total_raw = float(raw_gene.sum())
    total_adj = float(adj_gene.sum())
    total_rem = float(rem_gene.sum())

    gene_names = _names(result.gene_names, "gene", raw.shape[0])
    cell_names = _names(result.cell_names, "cell", raw.shape[1])

    per_gene = pd.DataFrame(
        {
            "gene": gene_names,
            "raw_counts": raw_gene,
            "adjusted_counts": adj_gene,
            "removed_counts": rem_gene,
            "removed_fraction": np.divide(rem_gene, raw_gene, out=np.zeros_like(rem_gene, dtype=float), where=raw_gene > 0),
            "share_of_all_removed": np.divide(rem_gene, total_rem, out=np.zeros_like(rem_gene, dtype=float), where=total_rem > 0),
        }
    ).sort_values(["removed_counts", "gene"], ascending=[False, True], ignore_index=True)
    per_gene["rank_removed"] = np.arange(1, len(per_gene) + 1)

    per_cell = pd.DataFrame(
        {
            "cell_id": cell_names,
            "raw_counts": raw_cell,
            "adjusted_counts": adj_cell,
            "removed_counts": rem_cell,
            "removed_fraction": np.divide(rem_cell, raw_cell, out=np.zeros_like(rem_cell, dtype=float), where=raw_cell > 0),
        }
    )
    if "status" in result.params.columns:
        status = result.params.copy()
        if "cell_index" in status.columns:
            status = status.sort_values("cell_index")
            per_cell["status"] = status["status"].to_numpy()
    per_cell = per_cell.sort_values(["removed_counts", "cell_id"], ascending=[False, True], ignore_index=True)
    per_cell["rank_removed"] = np.arange(1, len(per_cell) + 1)

    status_counts = (
        result.params["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n_cells")
        if "status" in result.params.columns
        else pd.DataFrame(columns=["status", "n_cells"])
    )
    if not status_counts.empty:
        status_counts["fraction_cells"] = status_counts["n_cells"] / raw.shape[1]

    nnz = raw.tocoo()
    removed_nnz = int(np.count_nonzero(removed[nnz.row, nnz.col].A1 > 0)) if nnz.nnz else 0
    summary_items = {
        "mode": (result.metadata or {}).get("mode", "unknown"),
        "raw_counts_total": total_raw,
        "adjusted_counts_total": total_adj,
        "removed_counts_total": total_rem,
        "removed_fraction": total_rem / total_raw if total_raw else 0.0,
        "n_cells": raw.shape[1],
        "n_genes": raw.shape[0],
        "nonzero_entries": int(raw.nnz),
        "nonzero_entries_removed": removed_nnz,
        "nonzero_entry_removed_fraction": removed_nnz / raw.nnz if raw.nnz else 0.0,
    }
    bg = (result.metadata or {}).get("background") or {}
    for key, value in bg.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary_items[f"background_{key}"] = value
    summary = pd.DataFrame({"metric": list(summary_items), "value": list(summary_items.values())})

    return {
        "summary": summary,
        "per_gene": per_gene,
        "per_cell": per_cell,
        "top_removed_genes": per_gene.head(top_n).copy(),
        "top_removed_cells": per_cell.head(top_n).copy(),
        "model_status": status_counts,
    }


def write_report_csv(
    report: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write report tables as a directory of CSV files."""

    out = Path(path)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out}")
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
    out.mkdir(parents=True, exist_ok=True)
    for name, table in report.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return out
