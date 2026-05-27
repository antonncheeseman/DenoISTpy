# DenoISTpy: sparse Python denoising for image-based spatial transcriptomics

## Overview

This repository contains `denoistpy`, an experimental Python implementation
of the DenoIST workflow for denoising image-based spatial transcriptomics data.
The Python package is designed around sparse count matrices, optional PyTorch
acceleration, and `SpatialData`/AnnData-style workflows such as Proseg outputs.

The original DenoIST implementation is an R/Bioconductor package. This
repository now carries the Python port/prototype under `denoistpy/`, which can
be installed with the `pyproject.toml`.

This is still a work in progress. The Python implementation currently focuses on
scalable data handling, Proseg/SpatialData ingestion, sparse local background
estimation, Poisson mixture fitting, AnnData/SpatialData writers, CLI usage, and
rough parity checks against the R implementation.

If you use this package and find it helpful/find issues, please let me know.
All credit for this denoising approach is attributed to the original implementation
by Aaron Kwok et al. (https://github.com/aaronkwc/DenoIST.git), the preprint for
which can be found at https://www.biorxiv.org/content/10.1101/2025.11.13.688387v1.full#T1

## What the Original R Package Does

The R package denoises image-based spatial transcriptomics count matrices by
estimating, for each cell, which gene counts are likely to be real cell signal
and which are likely contamination from local or ambient background.

At a high level, the R workflow:

1. Takes a genes x cells count matrix, transcript coordinates, and cell centroid
   coordinates. Inputs can come from a `SpatialExperiment` object or separate
   matrix/data-frame inputs.
2. Estimates ambient per-gene background from transcript locations by binning
   transcripts spatially with `hexbin`, filtering low-quality transcripts when a
   QV column is available, and fitting a mixture model to identify background
   bins.
3. Estimates local neighbourhood exposure by finding nearby cells within a
   distance radius. The newer fast path uses `dbscan` for this neighbour search.
4. For each cell, fits a two-component Poisson mixture model with `flexmix`,
   comparing observed gene counts against local/background offsets.
5. Returns adjusted counts, a genes x cells membership matrix where `1` means
   retained signal and `0` means inferred contamination, and fitted model
   parameters/posteriors.

The Python package keeps the same conceptual pipeline, but changes the
implementation details to be more scalable for high-plex/high-cell-count data:
sparse matrices for counts and neighbourhood aggregation, chunked dense batches
only for model fitting, optional PyTorch execution, and native AnnData/
SpatialData outputs.

## Python Installation

From this repository:

```bash
python -m pip install --editable ".[spatialdata,test]"
```

For GPU acceleration, install a platform-appropriate PyTorch build first or use
the optional `gpu` extra when suitable for your environment.

Tested during development with:

- Python 3.12.13
- PyTorch 2.11.0+cu128
- CUDA-enabled PyTorch wheel from the CUDA 12.8 PyTorch index

## Python Quick Start

```python
from denoistpy import denoist, from_proseg_spatialdata

inp = from_proseg_spatialdata(
    sdata,
    table_key="table",
    points_key="transcripts",
    transcript_position="adjusted", # or "observed" for observed_x/observed_y
    background_filter=None          # None, "exclude", or "only"
)

result = denoist(
    inp,
    x_col="x",
    y_col="y",
    gene_col="gene",
    distance=50,
    nbins=200,
    backend="torch",        # use "numpy" if PyTorch is unavailable
    device="auto",          # resolves to cuda, mps, or cpu
    batch_size=1024,
    store_memberships=False, # avoids a dense genes x cells debug matrix
    store_posterior=False,
    progress="auto"          # text in CLI-like sessions, tqdm in notebooks
)
```

The main result contains:

1. `adjusted_counts`: sparse genes x cells corrected counts.
2. `memberships`: optional genes x cells retained/removed indicators.
3. `params`: per-cell fitted model parameters and status flags.
4. `metadata`: run settings and background-estimation diagnostics.

For command-line or batch workflows:

```powershell
denoistpy run-proseg input.zarr output.zarr `
  --output-format spatialdata-zarr `
  --report-dir denoist_report `
  --table-key table `
  --points-key transcripts `
  --transcript-position adjusted `
  --distance 50 `
  --nbins 200 `
  --backend torch `
  --device auto `
  --batch-size 1024 `
  --progress text `
  --overwrite
```

Available output formats are `spatialdata-zarr`, `h5ad`, and `anndata-zarr`.
Use `--background-only` for ambient background subtraction without the Poisson
mixture model, and `--store-memberships` / `--store-posterior` only for smaller
or debugging runs because those outputs are dense genes x cells matrices.

## Python Implementation Notes

By default, local neighbourhood offsets include each cell's own counts once via
the radius-neighbour adjacency. The R fast path appears to include self in the
adjacency and then add the original count matrix again, effectively counting
self twice. The Python default is the cleaner `include_self_twice=False`;
set `include_self_twice=True` only when explicitly checking compatibility with
that R fast-path behaviour.

For a simpler ambient background subtraction path that skips the Poisson
mixture model:

```python
result = denoist(
    inp,
    distance=50,
    nbins=200,
    background_only=True
)
```

This mode estimates the per-gene ambient background from transcript hex bins,
subtracts it from each nonzero count with a floor at zero, and marks cells in
`params["status"]` as `background_only` or `zero_count`.

Results can also be exported back to Python spatial omics containers:

```python
from denoistpy import add_result_table_to_spatialdata, result_to_anndata

adata = result_to_anndata(
    result,
    raw_counts=inp,
    template_adata=inp.source_table,
    x="corrected"
)

sdata_with_denoist = add_result_table_to_spatialdata(
    sdata,
    result,
    raw_counts=inp,
    table_key="denoist",
    template_adata=inp.source_table
)
```

The AnnData export stores corrected counts in `X` by default, raw counts in
`layers["raw_counts"]`, corrected counts in `layers["denoist_corrected"]`,
per-cell model parameters in `obs`, and run metadata in `uns["denoist"]`.
When a template AnnData table is supplied, original `obs`, `var`, `obsm`, and
optionally `uns` metadata can be preserved in the exported table.

QC reports can be generated from raw and adjusted counts:

```python
from denoistpy import summarize_denoist_result, write_report_csv

report = summarize_denoist_result(result, raw_counts=inp, top_n=25)
write_report_csv(report, "denoist_report", overwrite=True)
```

The report includes summary metrics, per-gene and per-cell removed count mass,
top removed genes/cells, model status counts, and background-estimation
diagnostics such as transcript filtering, gene matching, occupied hex bins, and
background-bin counts. AnnData exports also add removed-count metrics to
`obs`, `var`, and `uns["denoist"]`.

## R/Python Parity Checks

The Python test suite includes optional rough parity checks against frozen CSV
references generated from the original R implementation. Run them with:

```powershell
python -m pytest tests_py/test_r_parity.py
```

The historical reference-regeneration script is still present under
`tests_py/parity/`, but it expects the original R package sources and fixtures
and is not currently wired into this Python-only repository layout.

The parity test uses fixed initialisation values and runs Python with
`include_self_twice=True` to approximate the current R fast-path behaviour.
The comparison is intentionally rough rather than bitwise exact because Python
uses native hex binning and `sklearn` GMM while R uses `hexbin` and `flexmix`.
