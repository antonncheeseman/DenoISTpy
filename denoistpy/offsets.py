from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture


@dataclass(slots=True)
class OffsetResult:
    """Sparse local offsets plus dense per-gene ambient background."""

    local: sparse.csr_matrix
    background: np.ndarray
    diagnostics: dict | None = None

    def dense_block(self, start: int, stop: int) -> np.ndarray:
        block = self.local[:, start:stop].toarray().astype(np.float32, copy=False)
        block += self.background[:, None]
        return block

    def to_sparse_with_background(self) -> sparse.csr_matrix:
        bg_matrix = sparse.csr_matrix(
            np.repeat(self.background[:, None], self.local.shape[1], axis=1)
        )
        return (self.local + bg_matrix).tocsr()


def filter_control_genes(
    counts: sparse.spmatrix,
    gene_names: np.ndarray | None,
    *,
    pattern: str = r"NegControl|BLANK|Unassigned",
) -> tuple[sparse.csr_matrix, np.ndarray | None]:
    """Drop control features following the R wrapper's default behavior."""

    counts = sparse.csr_matrix(counts)
    if gene_names is None:
        return counts, gene_names
    keep = np.asarray([re.search(pattern, str(g)) is None for g in gene_names])
    return counts[keep, :].tocsr(), gene_names[keep]


def _prepare_transcripts(
    transcripts: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    gene_col: str,
    qv_col: str | None,
    qv_threshold: float,
) -> pd.DataFrame:
    required = {x_col, y_col, gene_col}
    missing = required.difference(transcripts.columns)
    if missing:
        raise ValueError(f"Transcript table is missing required columns: {sorted(missing)}")
    tx = transcripts.loc[:, list(required) + ([qv_col] if qv_col in transcripts.columns else [])].copy()
    if qv_col is not None and qv_col in tx.columns:
        tx = tx.loc[tx[qv_col] >= qv_threshold]
    return tx


def _round_axial_hex(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round fractional axial hex coordinates to nearest integer hex."""

    x = q
    z = r
    y = -x - z

    rx = np.rint(x)
    ry = np.rint(y)
    rz = np.rint(z)

    x_diff = np.abs(rx - x)
    y_diff = np.abs(ry - y)
    z_diff = np.abs(rz - z)

    fix_x = (x_diff > y_diff) & (x_diff > z_diff)
    fix_y = (~fix_x) & (y_diff > z_diff)
    fix_z = ~(fix_x | fix_y)

    rx[fix_x] = -ry[fix_x] - rz[fix_x]
    ry[fix_y] = -rx[fix_y] - rz[fix_y]
    rz[fix_z] = -rx[fix_z] - ry[fix_z]
    return rx.astype(np.int64), rz.astype(np.int64)


def _hex_gene_bin_matrix(
    tx: pd.DataFrame,
    gene_to_idx: dict[object, int],
    *,
    x_col: str,
    y_col: str,
    gene_col: str,
    nbins: int,
) -> tuple[sparse.csr_matrix, float]:
    """Build a sparse gene x spatial-bin matrix.

    The assignment uses pointy-top axial hex coordinates and stores only
    occupied bins, which mirrors the behavior needed from R's ``hexbin(...,
    IDs = TRUE)`` for background estimation.
    """

    x = tx[x_col].to_numpy(dtype=float)
    y = tx[y_col].to_numpy(dtype=float)
    if x.size == 0:
        raise ValueError("No transcripts remain after filtering.")

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min = float(np.min(y))
    hex_radius = max((x_max - x_min) / max(nbins, 1) / math.sqrt(3), np.finfo(float).eps)
    hex_area = (3 * math.sqrt(3) / 2) * hex_radius**2

    # Convert to a local coordinate system before axial conversion. This keeps
    # bin ids compact and independent of absolute slide coordinates.
    x_local = x - x_min
    y_local = y - y_min
    q_frac = (math.sqrt(3) / 3 * x_local - 1 / 3 * y_local) / hex_radius
    r_frac = (2 / 3 * y_local) / hex_radius
    q, r = _round_axial_hex(q_frac, r_frac)
    hex_coords = pd.MultiIndex.from_arrays([q, r])
    bin_idx = pd.factorize(hex_coords, sort=False)[0].astype(np.int64)

    gene_idx = tx[gene_col].map(gene_to_idx).to_numpy()
    keep = pd.notna(gene_idx)
    gene_idx = gene_idx[keep].astype(np.int64)
    bin_idx = bin_idx[keep]

    data = np.ones(gene_idx.size, dtype=np.float32)
    mat = sparse.coo_matrix(
        (data, (gene_idx, bin_idx)),
        shape=(len(gene_to_idx), int(bin_idx.max(initial=-1)) + 1),
    ).tocsr()
    return mat, hex_area


def estimate_background_offset_with_diagnostics(
    transcripts: pd.DataFrame,
    gene_names: np.ndarray,
    *,
    x_col: str = "x",
    y_col: str = "y",
    gene_col: str = "gene",
    qv_col: str | None = "qv",
    qv_threshold: float = 20,
    distance: float = 50,
    nbins: int = 200,
    random_state: int | None = 0,
) -> tuple[np.ndarray, dict]:
    """Estimate per-gene ambient/background offset and diagnostics."""

    n_transcripts_input = len(transcripts)
    tx = _prepare_transcripts(
        transcripts,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
        qv_col=qv_col,
        qv_threshold=qv_threshold,
    )
    n_transcripts_used = len(tx)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    tx_genes = pd.Index(pd.unique(tx[gene_col]))
    count_genes = pd.Index(gene_names)
    matched_genes = tx_genes.intersection(count_genes)
    gene_bin_matrix, bin_area = _hex_gene_bin_matrix(
        tx,
        gene_to_idx,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
        nbins=nbins,
    )

    bin_total = np.asarray(gene_bin_matrix.sum(axis=0)).ravel()
    occupied_bins = bin_total > 0
    diagnostics = {
        "n_transcripts_input": int(n_transcripts_input),
        "n_transcripts_used": int(n_transcripts_used),
        "n_transcripts_filtered": int(n_transcripts_input - n_transcripts_used),
        "qv_col": qv_col,
        "qv_threshold": qv_threshold,
        "n_count_genes": int(len(count_genes)),
        "n_transcript_genes": int(len(tx_genes)),
        "n_matched_genes": int(len(matched_genes)),
        "n_unmatched_transcript_genes": int(len(tx_genes.difference(count_genes))),
        "n_count_genes_without_transcripts": int(len(count_genes.difference(tx_genes))),
        "n_hex_bins": int(gene_bin_matrix.shape[1]),
        "n_occupied_hex_bins": int(np.count_nonzero(occupied_bins)),
        "hex_area": float(bin_area),
        "background_method": "gmm",
        "gmm_means": None,
        "n_background_bins": 0,
    }
    if np.count_nonzero(occupied_bins) < 2:
        diagnostics["background_method"] = "ones_too_few_bins"
        return np.ones(len(gene_names), dtype=np.float32), diagnostics

    occupied_totals = bin_total[occupied_bins]
    try:
        gmm = GaussianMixture(n_components=2, random_state=random_state)
        occupied_labels = gmm.fit_predict(occupied_totals.reshape(-1, 1))
        background_component = int(np.argmin(gmm.means_.ravel()))
        background_bins = np.zeros_like(occupied_bins, dtype=bool)
        background_bins[occupied_bins] = occupied_labels == background_component
        diagnostics["gmm_means"] = [float(x) for x in gmm.means_.ravel()]
    except Exception:
        background_bins = occupied_bins & (bin_total <= np.quantile(occupied_totals, 0.25))
        diagnostics["background_method"] = "quantile_fallback"
    if not np.any(background_bins):
        diagnostics["background_method"] = "ones_no_background_bins"
        return np.ones(len(gene_names), dtype=np.float32), diagnostics

    diagnostics["n_background_bins"] = int(np.count_nonzero(background_bins))
    empty_gene_counts = np.asarray(gene_bin_matrix[:, background_bins].sum(axis=1)).ravel()
    per_unit = empty_gene_counts / (float(np.count_nonzero(background_bins)) * bin_area)
    scaled = per_unit * math.pi * distance**2
    bg = np.ceil(scaled).astype(np.float32)
    bg[bg == 0] = 1
    diagnostics["background_total"] = float(np.sum(bg))
    diagnostics["background_max"] = float(np.max(bg)) if bg.size else 0.0
    return bg, diagnostics


def estimate_background_offset(*args, **kwargs) -> np.ndarray:
    """Estimate per-gene ambient/background offset from transcript positions."""

    bg, _ = estimate_background_offset_with_diagnostics(*args, **kwargs)
    return bg


def radius_adjacency(coords: np.ndarray, distance: float, *, include_self: bool = True) -> sparse.csr_matrix:
    """Build a sparse cells x cells radius-neighbor adjacency matrix."""

    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords[:, :2])
    neighbors = tree.query_ball_point(coords[:, :2], r=distance)

    rows: list[int] = []
    cols: list[int] = []
    for cell_idx, ids in enumerate(neighbors):
        for neighbor_idx in ids:
            if include_self or neighbor_idx != cell_idx:
                rows.append(neighbor_idx)
                cols.append(cell_idx)

    data = np.ones(len(rows), dtype=np.float32)
    n_cells = coords.shape[0]
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells)).tocsr()


def compute_sparse_local_offsets(
    counts: sparse.spmatrix,
    coords: np.ndarray,
    transcripts: pd.DataFrame,
    *,
    gene_names: np.ndarray | None = None,
    x_col: str = "x",
    y_col: str = "y",
    gene_col: str = "gene",
    qv_col: str | None = "qv",
    qv_threshold: float = 20,
    distance: float = 50,
    nbins: int = 200,
    include_self_twice: bool = False,
    random_state: int | None = 0,
) -> OffsetResult:
    """Compute sparse local offsets plus ambient background.

    The radius adjacency includes each cell itself once. By default we do not
    add self counts again, so the local offset is neighbors plus self plus
    ambient background. Set ``include_self_twice=True`` only for compatibility
    checks against the current R fast implementation, which appears to include
    self in the adjacency and then add the original count matrix again.
    """

    counts = sparse.csr_matrix(counts, dtype=np.float32)
    if gene_names is None:
        gene_names = np.arange(counts.shape[0])
    bg, diagnostics = estimate_background_offset_with_diagnostics(
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

    adjacency = radius_adjacency(coords, distance, include_self=True)
    offsets = (counts @ adjacency).tocsr()
    if include_self_twice:
        offsets = offsets + counts

    return OffsetResult(local=offsets.tocsr(), background=bg, diagnostics=diagnostics)


def local_offset_distance_with_background(
    counts: sparse.spmatrix,
    coords: np.ndarray,
    transcripts: pd.DataFrame,
    **kwargs,
) -> sparse.csr_matrix:
    """Compatibility wrapper returning a matrix with background materialized.

    For large datasets prefer :func:`compute_sparse_local_offsets`, which keeps
    ambient background as a per-gene vector instead of expanding it across every
    cell.
    """

    return compute_sparse_local_offsets(counts, coords, transcripts, **kwargs).to_sparse_with_background()
