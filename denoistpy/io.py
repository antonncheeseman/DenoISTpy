from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(slots=True)
class DenoistInput:
    """Canonical input representation for the Python DenoIST pipeline."""

    counts: sparse.spmatrix
    coords: np.ndarray
    transcripts: pd.DataFrame
    gene_names: np.ndarray | None = None
    cell_names: np.ndarray | None = None
    source_table: Any | None = None


def _as_sparse_genes_by_cells(x: Any, *, from_cells_by_genes: bool) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(x)
    if from_cells_by_genes:
        matrix = matrix.T.tocsr()
    return matrix


def from_anndata(
    adata: Any,
    transcripts: pd.DataFrame,
    *,
    layer: str | None = None,
    spatial_key: str = "spatial",
    cell_coordinate_source: str = "obsm",
    cell_x_col: str | None = None,
    cell_y_col: str | None = None,
    gene_names_col: str | None = None,
) -> DenoistInput:
    """Create a :class:`DenoistInput` from an AnnData object.

    AnnData stores expression as cells x genes, so the returned count matrix is
    transposed to genes x cells.
    """

    x = adata.layers[layer] if layer is not None else adata.X
    if cell_coordinate_source == "obsm":
        coords = np.asarray(adata.obsm[spatial_key])
    elif cell_coordinate_source == "obs":
        if cell_x_col is None or cell_y_col is None:
            raise ValueError("cell_x_col and cell_y_col are required when cell_coordinate_source='obs'.")
        coords = adata.obs.loc[:, [cell_x_col, cell_y_col]].to_numpy()
    else:
        raise ValueError("cell_coordinate_source must be either 'obsm' or 'obs'.")

    gene_names = np.asarray(adata.var[gene_names_col]) if gene_names_col is not None else np.asarray(adata.var_names)
    return DenoistInput(
        counts=_as_sparse_genes_by_cells(x, from_cells_by_genes=True),
        coords=coords,
        transcripts=transcripts,
        gene_names=gene_names,
        cell_names=np.asarray(adata.obs_names),
        source_table=adata,
    )


def _first_mapping_value(mapping: Any, key: str | None) -> Any:
    if key is not None:
        return mapping[key]
    if not mapping:
        raise ValueError("No matching SpatialData element was found.")
    return next(iter(mapping.values()))


def _materialize_dataframe(obj: Any, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is not None:
        try:
            obj = obj.loc[:, columns]
        except Exception:
            obj = obj[columns]
    if hasattr(obj, "compute"):
        obj = obj.compute()
    if hasattr(obj, "to_pandas"):
        obj = obj.to_pandas()
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj)


def from_spatialdata(
    sdata: Any,
    *,
    table_key: str | None = None,
    points_key: str | None = None,
    layer: str | None = None,
    spatial_key: str = "spatial",
    transcript_columns: list[str] | None = None,
    cell_coordinate_source: str = "obsm",
    cell_x_col: str | None = None,
    cell_y_col: str | None = None,
    gene_names_col: str | None = None,
) -> DenoistInput:
    """Create a :class:`DenoistInput` from a SpatialData object.

    This intentionally accepts duck-typed SpatialData-like objects so the core
    package does not need to import ``spatialdata`` unless users install it.
    The function expects an AnnData-like table plus a points element containing
    transcript records.
    """

    table = _first_mapping_value(sdata.tables, table_key)
    points = _first_mapping_value(sdata.points, points_key)
    transcripts = _materialize_dataframe(points, transcript_columns)
    return from_anndata(
        table,
        transcripts,
        layer=layer,
        spatial_key=spatial_key,
        cell_coordinate_source=cell_coordinate_source,
        cell_x_col=cell_x_col,
        cell_y_col=cell_y_col,
        gene_names_col=gene_names_col,
    )


def from_proseg_spatialdata(
    sdata: Any,
    *,
    table_key: str = "table",
    points_key: str = "transcripts",
    layer: str | None = None,
    transcript_position: str = "adjusted",
    x_col: str | None = None,
    y_col: str | None = None,
    gene_col: str = "gene",
    background_col: str = "background",
    background_filter: str | None = None,
    spatial_key: str = "spatial",
    cell_coordinate_source: str = "obsm",
    cell_x_col: str | None = None,
    cell_y_col: str | None = None,
    gene_names_col: str | None = None,
) -> DenoistInput:
    """Create a :class:`DenoistInput` from Proseg-style SpatialData output.

    Proseg transcript points commonly expose both adjusted positions
    (``x``/``y``) and observed raw positions (``observed_x``/``observed_y``).
    Explicit ``x_col``/``y_col`` values override ``transcript_position``.

    ``background_filter`` controls use of Proseg's transcript-level background
    flag: ``None`` keeps all transcripts, ``"exclude"`` removes background
    transcripts, and ``"only"`` keeps only background transcripts.
    """

    if transcript_position == "adjusted":
        default_x, default_y = "x", "y"
    elif transcript_position == "observed":
        default_x, default_y = "observed_x", "observed_y"
    else:
        raise ValueError("transcript_position must be either 'adjusted' or 'observed'.")

    x_col = x_col or default_x
    y_col = y_col or default_y
    if background_filter not in {None, "exclude", "only"}:
        raise ValueError("background_filter must be None, 'exclude', or 'only'.")

    transcript_columns = [x_col, y_col, gene_col]
    if background_filter is not None:
        transcript_columns.append(background_col)

    table = _first_mapping_value(sdata.tables, table_key)
    points = _first_mapping_value(sdata.points, points_key)
    transcripts = _materialize_dataframe(points, transcript_columns)
    if background_filter == "exclude":
        transcripts = transcripts.loc[~transcripts[background_col].astype(bool)].copy()
    elif background_filter == "only":
        transcripts = transcripts.loc[transcripts[background_col].astype(bool)].copy()

    if x_col != "x":
        transcripts = transcripts.rename(columns={x_col: "x"})
        x_col = "x"
    if y_col != "y":
        transcripts = transcripts.rename(columns={y_col: "y"})
        y_col = "y"

    return from_anndata(
        table,
        transcripts,
        layer=layer,
        spatial_key=spatial_key,
        cell_coordinate_source=cell_coordinate_source,
        cell_x_col=cell_x_col,
        cell_y_col=cell_y_col,
        gene_names_col=gene_names_col,
    )
