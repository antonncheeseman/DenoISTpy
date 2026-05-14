from __future__ import annotations

from collections.abc import Mapping
from copy import copy, deepcopy
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .io import DenoistInput
from .report import summarize_denoist_result
from .types import DenoistResult


def _prepare_output_path(path: str | Path, *, overwrite: bool) -> Path:
    out = Path(path)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out}")
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _require_anndata() -> type:
    try:
        from anndata import AnnData
    except ImportError as exc:
        raise ImportError(
            "AnnData output requires the optional 'anndata' dependency. "
            "Install with `pip install denoistpy[spatialdata]`."
        ) from exc
    return AnnData


def _coerce_counts(
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    *,
    result: DenoistResult,
) -> sparse.csr_matrix:
    if isinstance(raw_counts, DenoistInput):
        raw_counts = raw_counts.counts
    counts = sparse.csr_matrix(raw_counts)
    if counts.shape != result.adjusted_counts.shape:
        raise ValueError(
            "raw_counts must be genes x cells and match result.adjusted_counts. "
            f"Got {counts.shape}, expected {result.adjusted_counts.shape}."
        )
    return counts


def _template_from_counts(raw_counts: sparse.spmatrix | np.ndarray | DenoistInput) -> Any | None:
    if isinstance(raw_counts, DenoistInput):
        return raw_counts.source_table
    return None


def _names_or_default(names: np.ndarray | None, prefix: str, n: int) -> pd.Index:
    if names is None:
        return pd.Index([f"{prefix}{idx}" for idx in range(n)])
    if len(names) != n:
        raise ValueError(f"Expected {n} {prefix} names, got {len(names)}.")
    return pd.Index([str(name) for name in names])


def _params_by_cell(result: DenoistResult, cell_index: pd.Index) -> pd.DataFrame:
    params = result.params.copy()
    if "cell_index" in params.columns:
        params = params.sort_values("cell_index")
        params.index = cell_index[params["cell_index"].to_numpy()]
    elif "cell_id" in params.columns:
        params.index = pd.Index(params["cell_id"].astype(str))
    else:
        params.index = cell_index[: len(params)]
    params = params.reindex(cell_index)
    params.index.name = None
    return params.add_prefix("denoist_")


def _obs_from_template(result: DenoistResult, cell_index: pd.Index, template_adata: Any | None) -> pd.DataFrame:
    if template_adata is not None:
        if template_adata.n_obs != len(cell_index):
            raise ValueError("template_adata must have the same number of cells as the DenoIST result.")
        obs = template_adata.obs.copy()
        obs.index = cell_index
    else:
        obs = pd.DataFrame(index=cell_index)
    params = _params_by_cell(result, cell_index)
    for col in params.columns:
        obs[col] = params[col]
    obs.index.name = None
    return obs


def _var_from_template(gene_index: pd.Index, template_adata: Any | None) -> pd.DataFrame:
    if template_adata is not None:
        if template_adata.n_vars != len(gene_index):
            raise ValueError("template_adata must have the same number of genes as the DenoIST result.")
        var = template_adata.var.copy()
        var.index = gene_index
        return var
    return pd.DataFrame(index=gene_index)


def _copy_axis_mappings(target: Any, template_adata: Any | None) -> None:
    if template_adata is None:
        return
    for key, value in template_adata.obsm.items():
        target.obsm[key] = value.copy() if hasattr(value, "copy") else deepcopy(value)
    for key, value in template_adata.obsp.items():
        target.obsp[key] = value.copy() if hasattr(value, "copy") else deepcopy(value)


def result_to_anndata(
    result: DenoistResult,
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    *,
    x: str = "corrected",
    raw_layer: str = "raw_counts",
    corrected_layer: str = "denoist_corrected",
    memberships_layer: str | None = "denoist_membership",
    posterior_layer: str | None = None,
    include_raw_slot: bool = True,
    template_adata: Any | None = None,
    copy_uns_from_template: bool = False,
    add_report_metrics: bool = True,
    uns_key: str = "denoist",
) -> Any:
    """Create an AnnData object containing raw and DenoIST-corrected counts.

    ``DenoistResult`` stores matrices as genes x cells. AnnData stores cells x
    genes, so all layers are transposed during export.
    """

    AnnData = _require_anndata()
    template_adata = template_adata or _template_from_counts(raw_counts)
    raw = _coerce_counts(raw_counts, result=result)
    corrected = sparse.csr_matrix(result.adjusted_counts)

    n_genes, n_cells = corrected.shape
    gene_index = _names_or_default(result.gene_names, "gene", n_genes)
    cell_index = _names_or_default(result.cell_names, "cell", n_cells)

    if x == "corrected":
        x_matrix = corrected.T.tocsr()
    elif x == "raw":
        x_matrix = raw.T.tocsr()
    else:
        raise ValueError("x must be either 'corrected' or 'raw'.")

    adata = AnnData(
        X=x_matrix,
        obs=_obs_from_template(result, cell_index, template_adata),
        var=_var_from_template(gene_index, template_adata),
    )
    _copy_axis_mappings(adata, template_adata)
    if copy_uns_from_template and template_adata is not None:
        adata.uns.update(deepcopy(dict(template_adata.uns)))

    adata.layers[raw_layer] = raw.T.tocsr()
    adata.layers[corrected_layer] = corrected.T.tocsr()

    if memberships_layer is not None and result.memberships is not None:
        adata.layers[memberships_layer] = sparse.csr_matrix(result.memberships).T.tocsr()
    if posterior_layer is not None and result.posterior is not None:
        adata.layers[posterior_layer] = sparse.csr_matrix(result.posterior).T.tocsr()
    if include_raw_slot:
        adata.raw = AnnData(
            X=raw.T.tocsr(),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )

    report = summarize_denoist_result(result, raw) if add_report_metrics else None
    if report is not None:
        per_cell = report["per_cell"].set_index("cell_id").reindex(adata.obs_names)
        per_gene = report["per_gene"].set_index("gene").reindex(adata.var_names)
        adata.obs["denoist_raw_counts"] = per_cell["raw_counts"].to_numpy()
        adata.obs["denoist_adjusted_counts"] = per_cell["adjusted_counts"].to_numpy()
        adata.obs["denoist_removed_counts"] = per_cell["removed_counts"].to_numpy()
        adata.obs["denoist_removed_fraction"] = per_cell["removed_fraction"].to_numpy()
        adata.var["denoist_raw_counts"] = per_gene["raw_counts"].to_numpy()
        adata.var["denoist_adjusted_counts"] = per_gene["adjusted_counts"].to_numpy()
        adata.var["denoist_removed_counts"] = per_gene["removed_counts"].to_numpy()
        adata.var["denoist_removed_fraction"] = per_gene["removed_fraction"].to_numpy()

    adata.uns[uns_key] = {
        "metadata": dict(result.metadata or {}),
        "matrix_orientation": "AnnData cells x genes; DenoIST internal matrices are genes x cells",
        "layers": {
            "raw": raw_layer,
            "corrected": corrected_layer,
            "memberships": memberships_layer,
            "posterior": posterior_layer,
        },
    }
    if report is not None:
        adata.uns[uns_key]["summary"] = dict(zip(report["summary"]["metric"], report["summary"]["value"]))
        adata.uns[uns_key]["model_status"] = report["model_status"].to_dict(orient="list")
    return adata


def add_result_table_to_spatialdata(
    sdata: Any,
    result: DenoistResult,
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    *,
    table_key: str = "denoist",
    copy_sdata: bool = True,
    copy_mode: str = "deep",
    **anndata_kwargs: Any,
) -> Any:
    """Return a SpatialData object with a DenoIST AnnData table added.

    The original object is copied by default, then ``tables[table_key]`` is set
    to an AnnData table containing raw counts, corrected counts, per-cell model
    parameters in ``obs``, and run metadata in ``uns``.
    """

    if copy_sdata:
        if copy_mode == "deep":
            out = deepcopy(sdata)
        elif copy_mode == "shallow":
            out = copy(sdata)
        else:
            raise ValueError("copy_mode must be either 'deep' or 'shallow'.")
    else:
        out = sdata

    if not hasattr(out, "tables"):
        raise TypeError("sdata must expose a 'tables' mapping.")
    if not isinstance(out.tables, Mapping) and not hasattr(out.tables, "__setitem__"):
        raise TypeError("sdata.tables must support item assignment.")

    out.tables[table_key] = result_to_anndata(result, raw_counts, **anndata_kwargs)
    return out


def write_result_h5ad(
    result: DenoistResult,
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    path: str | Path,
    *,
    overwrite: bool = False,
    compression: str | None = None,
    compression_opts: int | Any | None = None,
    **anndata_kwargs: Any,
) -> Path:
    """Write a DenoIST result as a single H5AD file."""

    out = _prepare_output_path(path, overwrite=overwrite)
    adata = result_to_anndata(result, raw_counts, **anndata_kwargs)
    adata.write_h5ad(
        out,
        compression=compression,
        compression_opts=compression_opts,
    )
    return out


def write_result_anndata_zarr(
    result: DenoistResult,
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    path: str | Path,
    *,
    overwrite: bool = False,
    chunks: tuple[int, ...] | None = None,
    **anndata_kwargs: Any,
) -> Path:
    """Write a DenoIST result as an AnnData Zarr store."""

    out = _prepare_output_path(path, overwrite=overwrite)
    adata = result_to_anndata(result, raw_counts, **anndata_kwargs)
    adata.write_zarr(out, chunks=chunks)
    return out


def write_result_spatialdata_zarr(
    sdata: Any,
    result: DenoistResult,
    raw_counts: sparse.spmatrix | np.ndarray | DenoistInput,
    path: str | Path,
    *,
    table_key: str = "denoist",
    overwrite: bool = False,
    copy_sdata: bool = True,
    copy_mode: str = "deep",
    consolidate_metadata: bool = True,
    update_sdata_path: bool = True,
    **anndata_kwargs: Any,
) -> Path:
    """Add a DenoIST table to SpatialData and write it as a Zarr store."""

    out = Path(path)
    sdata_out = add_result_table_to_spatialdata(
        sdata,
        result,
        raw_counts,
        table_key=table_key,
        copy_sdata=copy_sdata,
        copy_mode=copy_mode,
        **anndata_kwargs,
    )
    if not hasattr(sdata_out, "write"):
        raise TypeError("sdata must expose a SpatialData-like write(path, overwrite=...) method.")
    out.parent.mkdir(parents=True, exist_ok=True)
    sdata_out.write(
        out,
        overwrite=overwrite,
        consolidate_metadata=consolidate_metadata,
        update_sdata_path=update_sdata_path,
    )
    return out
