"""Sparse-first Python prototype for the DenoIST pipeline."""

from .core import denoist
from .export import (
    add_result_table_to_spatialdata,
    result_to_anndata,
    write_result_anndata_zarr,
    write_result_h5ad,
    write_result_spatialdata_zarr,
)
from .io import DenoistInput, from_anndata, from_proseg_spatialdata, from_spatialdata
from .mixture import solve_poisson_mixture_numpy, solve_poisson_mixture_torch
from .offsets import compute_sparse_local_offsets, local_offset_distance_with_background
from .progress import ProgressReporter
from .report import summarize_denoist_result, write_report_csv
from .types import DenoistResult

__all__ = [
    "DenoistInput",
    "DenoistResult",
    "denoist",
    "result_to_anndata",
    "add_result_table_to_spatialdata",
    "write_result_h5ad",
    "write_result_anndata_zarr",
    "write_result_spatialdata_zarr",
    "summarize_denoist_result",
    "write_report_csv",
    "from_anndata",
    "from_proseg_spatialdata",
    "from_spatialdata",
    "compute_sparse_local_offsets",
    "local_offset_distance_with_background",
    "solve_poisson_mixture_numpy",
    "solve_poisson_mixture_torch",
    "ProgressReporter",
]
