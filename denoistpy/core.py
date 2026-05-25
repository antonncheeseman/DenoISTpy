from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .io import DenoistInput
from .mixture import solve_poisson_mixture_numpy, solve_poisson_mixture_torch
from .offsets import (
    OffsetResult,
    compute_sparse_local_offsets,
    estimate_background_offset_with_diagnostics,
    filter_control_genes,
)
from .progress import ProgressMode, ProgressReporter
from .types import DenoistResult


def _normalize_input(
    counts: sparse.spmatrix | DenoistInput,
    transcripts: pd.DataFrame | None,
    coords: np.ndarray | None,
    gene_names: np.ndarray | None,
    cell_names: np.ndarray | None,
) -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    if isinstance(counts, DenoistInput):
        inp = counts
        return (
            sparse.csr_matrix(inp.counts),
            np.asarray(inp.coords),
            inp.transcripts,
            inp.gene_names,
            inp.cell_names,
        )
    if transcripts is None:
        raise ValueError("transcripts must be provided unless counts is a DenoistInput.")
    if coords is None:
        raise ValueError("coords must be provided unless counts is a DenoistInput.")
    return sparse.csr_matrix(counts), np.asarray(coords), transcripts, gene_names, cell_names


def _batched_em(
    counts: sparse.spmatrix,
    offsets: OffsetResult,
    *,
    backend: str,
    batch_size: int,
    posterior_cutoff: float,
    n_inits: int | np.ndarray,
    max_iter: int,
    tol: float,
    device: str,
    store_memberships: bool,
    store_posterior: bool,
    random_state: int | None,
    progress: ProgressReporter,
) -> tuple[sparse.csr_matrix, np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    n_genes, n_cells = counts.shape
    memberships = np.ones((n_genes, n_cells), dtype=np.int8) if store_memberships else None
    posterior = np.ones((n_genes, n_cells), dtype=np.float32) if store_posterior else None
    params: list[pd.DataFrame] = []
    adjusted_rows: list[np.ndarray] = []
    adjusted_cols: list[np.ndarray] = []
    adjusted_data: list[np.ndarray] = []

    counts_csc = counts.tocsc()
    offsets_csc = offsets.local.tocsc()

    starts = range(0, n_cells, batch_size)
    n_batches = (n_cells + batch_size - 1) // batch_size
    for start in progress.iter_batches(
        starts,
        total=n_batches,
        label="EM batches",
        batch_size=batch_size,
        n_items=n_cells,
    ):
        stop = min(start + batch_size, n_cells)
        x = counts_csc[:, start:stop].toarray().astype(np.float32, copy=False)
        s = offsets_csc[:, start:stop].toarray().astype(np.float32, copy=False)
        s += offsets.background[:, None]
        seed = None if random_state is None else random_state + start

        if backend == "torch":
            batch = solve_poisson_mixture_torch(
                x,
                s,
                max_iter=max_iter,
                tol=tol,
                n_inits=n_inits,
                posterior_cutoff=posterior_cutoff,
                device=device,
                random_state=seed,
            )
        elif backend == "numpy":
            columns = [
                solve_poisson_mixture_numpy(
                    x[:, idx],
                    s[:, idx],
                    max_iter=max_iter,
                    tol=tol,
                    n_inits=n_inits,
                    posterior_cutoff=posterior_cutoff,
                    random_state=None if seed is None else seed + idx,
                )
                for idx in range(stop - start)
            ]
            batch = {
                "memberships": np.column_stack([col["memberships"] for col in columns]),
                "posterior": np.column_stack([col["posterior"] for col in columns]),
                "lambda1": np.asarray([col["lambda1"] for col in columns]),
                "lambda2": np.asarray([col["lambda2"] for col in columns]),
                "pi": np.asarray([col["pi"] for col in columns]),
                "log_lik": np.asarray([col["log_lik"] for col in columns]),
                "n_iter": np.asarray([col["n_iter"] for col in columns]),
                "status": np.asarray([col["status"] for col in columns], dtype=object),
            }
        else:
            raise ValueError("backend must be either 'numpy' or 'torch'.")

        if memberships is not None:
            memberships[:, start:stop] = batch["memberships"]
        if posterior is not None:
            posterior[:, start:stop] = batch["posterior"]

        counts_block = counts_csc[:, start:stop].tocoo()
        keep_counts = batch["memberships"][counts_block.row, counts_block.col].astype(bool)
        adjusted_rows.append(counts_block.row[keep_counts])
        adjusted_cols.append(counts_block.col[keep_counts] + start)
        adjusted_data.append(counts_block.data[keep_counts])

        params.append(
            pd.DataFrame(
                {
                    "cell_index": np.arange(start, stop),
                    "lambda1": batch["lambda1"],
                    "lambda2": batch["lambda2"],
                    "pi": batch["pi"],
                    "log_lik": batch["log_lik"],
                    "n_iter": batch["n_iter"],
                    "status": batch["status"],
                }
            )
        )

    if adjusted_data:
        adjusted = sparse.coo_matrix(
            (
                np.concatenate(adjusted_data),
                (np.concatenate(adjusted_rows), np.concatenate(adjusted_cols)),
            ),
            shape=counts.shape,
        ).tocsr()
    else:
        adjusted = sparse.csr_matrix(counts.shape, dtype=counts.dtype)

    return adjusted, memberships, posterior, pd.concat(params, ignore_index=True)


def _background_only_adjustment(
    counts: sparse.spmatrix,
    background: np.ndarray,
    *,
    store_memberships: bool,
    store_posterior: bool,
) -> tuple[sparse.csr_matrix, np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    coo = counts.tocoo()
    adjusted_values = coo.data.astype(np.float32, copy=True) - background[coo.row]
    keep = adjusted_values > 0
    adjusted = sparse.coo_matrix(
        (adjusted_values[keep], (coo.row[keep], coo.col[keep])),
        shape=counts.shape,
    ).tocsr()

    memberships = None
    if store_memberships:
        memberships = np.ones(counts.shape, dtype=np.int8)
        memberships[coo.row[~keep], coo.col[~keep]] = 0

    posterior = np.ones(counts.shape, dtype=np.float32) if store_posterior else None
    cell_sums = np.asarray(counts.sum(axis=0)).ravel()
    status = np.where(cell_sums == 0, "zero_count", "background_only")
    params = pd.DataFrame(
        {
            "cell_index": np.arange(counts.shape[1]),
            "lambda1": np.nan,
            "lambda2": np.nan,
            "pi": np.nan,
            "log_lik": np.nan,
            "n_iter": 0,
            "status": status,
        }
    )
    return adjusted, memberships, posterior, params


def denoist(
    counts: sparse.spmatrix | DenoistInput,
    transcripts: pd.DataFrame | None = None,
    coords: np.ndarray | None = None,
    *,
    gene_names: np.ndarray | None = None,
    cell_names: np.ndarray | None = None,
    x_col: str = "x",
    y_col: str = "y",
    gene_col: str = "gene",
    qv_col: str | None = "qv",
    qv_threshold: float = 20,
    distance: float = 50,
    nbins: int = 200,
    posterior_cutoff: float = 0.6,
    n_inits: int | np.ndarray = 10,
    max_iter: int = 5000,
    tol: float = 1e-6,
    backend: str = "numpy",
    device: str = "auto",
    batch_size: int = 1024,
    store_memberships: bool = True,
    store_posterior: bool = False,
    background_only: bool = False,
    include_self_twice: bool = False,
    return_offsets: bool = False,
    random_state: int | None = 0,
    progress: ProgressMode | bool | None = None,
) -> DenoistResult:
    """Run the Python DenoIST prototype.

    Inputs and outputs are genes x cells internally. Use :func:`from_anndata`
    or :func:`from_spatialdata` for common spatial omics containers.
    """

    progress_reporter = ProgressReporter(progress)
    with progress_reporter.phase("Preparing input"):
        counts, coords, transcripts, gene_names, cell_names = _normalize_input(
            counts,
            transcripts,
            coords,
            gene_names,
            cell_names,
        )
        counts, gene_names = filter_control_genes(counts, gene_names)
    if gene_names is None:
        gene_names = np.arange(counts.shape[0])

    if background_only:
        with progress_reporter.phase("Estimating ambient background"):
            background, background_diagnostics = estimate_background_offset_with_diagnostics(
                transcripts,
                np.asarray(gene_names),
                x_col=x_col,
                y_col=y_col,
                gene_col=gene_col,
                qv_col=qv_col,
                qv_threshold=qv_threshold,
                distance=distance,
                nbins=nbins,
                random_state=random_state,
            )
        with progress_reporter.phase("Applying background-only adjustment"):
            adjusted, memberships, posterior, params = _background_only_adjustment(
                counts,
                background,
                store_memberships=store_memberships,
                store_posterior=store_posterior,
            )
        if cell_names is not None:
            params.insert(1, "cell_id", cell_names[params["cell_index"].to_numpy()])
        offsets = None
        if return_offsets:
            offsets = sparse.csr_matrix(np.repeat(background[:, None], counts.shape[1], axis=1))
        return DenoistResult(
            adjusted_counts=adjusted,
            memberships=memberships,
            posterior=posterior,
            params=params,
            offsets=offsets,
            gene_names=gene_names,
            cell_names=cell_names,
            metadata={
                "mode": "background_only",
                "backend": None,
                "distance": distance,
                "nbins": nbins,
                "background_only": True,
                "background": background_diagnostics,
            },
        )

    with progress_reporter.phase("Computing local/background offsets"):
        offsets = compute_sparse_local_offsets(
            counts,
            coords,
            transcripts,
            gene_names=gene_names,
            x_col=x_col,
            y_col=y_col,
            gene_col=gene_col,
            qv_col=qv_col,
            qv_threshold=qv_threshold,
            distance=distance,
            nbins=nbins,
            include_self_twice=include_self_twice,
            random_state=random_state,
        )

    with progress_reporter.phase("Fitting Poisson mixture", f"{counts.shape[1]} cells in batches of {batch_size}"):
        adjusted, memberships, posterior, params = _batched_em(
            counts,
            offsets,
            backend=backend,
            batch_size=batch_size,
            posterior_cutoff=posterior_cutoff,
            n_inits=n_inits,
            max_iter=max_iter,
            tol=tol,
            device=device,
            store_memberships=store_memberships,
            store_posterior=store_posterior,
            random_state=random_state,
            progress=progress_reporter,
        )
    if cell_names is not None:
        params.insert(1, "cell_id", cell_names[params["cell_index"].to_numpy()])

    return DenoistResult(
        adjusted_counts=adjusted,
        memberships=memberships,
        posterior=posterior,
        params=params,
        offsets=offsets.to_sparse_with_background() if return_offsets else None,
        gene_names=gene_names,
        cell_names=cell_names,
        metadata={
            "backend": backend,
            "mode": "poisson_mixture",
            "distance": distance,
            "nbins": nbins,
            "posterior_cutoff": posterior_cutoff,
            "batch_size": batch_size,
            "include_self_twice": include_self_twice,
            "background_only": False,
            "background": offsets.diagnostics,
        },
    )
