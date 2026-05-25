# DenoISTpy: sparse Python denoising for image-based spatial transcriptomics

## Overview

This repository contains `denoistpy`, an experimental Python implementation
of the DenoIST workflow for denoising image-based spatial transcriptomics data.
The Python package is designed around sparse count matrices, optional PyTorch
acceleration, and `SpatialData`/AnnData-style workflows such as Proseg outputs.

The original DenoIST implementation is an R/Bioconductor package. The R code is
still present in this repository, but the Python prototype lives under
`denoistpy/` and can be installed with the `pyproject.toml` in this branch.

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

The Python test suite includes an optional R parity workflow using conda-managed
R. Create the R environment and generate references with:

```powershell
conda env create -f tests_py/parity/environment-r.yml
conda run -n denoistpy_r Rscript tests_py/parity/generate_r_reference.R
python -m pytest tests_py/test_r_parity.py
```

The parity test uses fixed initialisation values and runs Python with
`include_self_twice=True` to approximate the current R fast-path behaviour.
The comparison is intentionally rough rather than bitwise exact because Python
uses native hex binning and `sklearn` GMM while R uses `hexbin` and `flexmix`.
The conda R reference script defines a small `rowSums2` compatibility shim so
the parity environment can be created on Windows, where the Bioconductor conda
package used by the R package is not always available.

## Original R Package Usage

The inherited R package is designed to be used with the
[SpatialExperiment](https://bioconductor.org/packages/release/bioc/html/SpatialExperiment.html)
class. If using a different format, it can also accept a matrix input with a
data frame of coordinates.

## R Package News:
- **2026-03-17**: Major update
  - Added `local_offset_distance_with_background_fast` function which uses `dbscan` for faster neighbour finding. Also fixed critical bug that causes wrong distance calculation.
  - `denoist()` now defaults to fast neighbour finding via the `neighbour_mode` option.

- **2026-02-24**: Minor update
  - Fixed minor bug where background offset cannot be calculated because an entire gene gets filtered out because of low qv. This should not change existing usage as the issue only arises in extremely small toy datasets.
  - `n_inits` can now be tuned in the `denoist()` function for speed.

- **2025-06-25**: Major update
  - Fixed memory usage for parallel processing. Feature is available only on linux/UNIX machines due to dependency on `parallel`.
  - Posterior cutoff is now a tunable parameter.
  - QOL changes including checking input types and error handling.

- **2025-05-20**: Initial release of DenoIST. The package is now available on and GitHub.

## R Installation:

From Bioconductor:

```{r install}
if(!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")
BiocManager::install("DenoIST")
```

Or from Github directly:

```         
BiocManager::install(c('sparseMatrixStats', 'SpatialExperiment','SummarizedExperiment'))
devtools::install_github("aaronkwc/DenoIST")
```

## R Quick Start:

In most cases, you will only need to use the `denoist()` wrapper function.

It takes 2-3 inputs:

1.  `mat` : SpatialExperiment object (with the counts in assay() slot) or a count matrix with genes as rows and cells as columns.
2.  `tx`: Transcript data frame (a data frame with each row being an individual transcript, with columns specifying each transcripts' coordinates and qv). If your transcript file is not from Xenium and has no qv score, you can set a dummy column of `qv = 20` for all transcripts. This workaround should not be needed in future updates.
3.  `coords`: If using a count matrix, a data frame (cells x 2) for each cell's centroid 2D coordinate.

The function will return a list with

1.  `adjusted_counts`: The adjusted counts matrix with contamination removed.
2.  `memberships`: A data frame with the inferred identity of each gene in each cell (1 for real or 0 for contamination).
3.  `params`: A list with the estimated parameters used in the model. The posterior probabilities of each gene being real or contamination can be found in `params$posterior_probs`, higher means more likely to be contamination.

You can run `?denoist` for more details on the extra parameters you can adjust.

## R Examples

With a SpatialExperiment object:

```         
library(DenoIST)
library(SpatialExperiment)

res <- denoist(mat = spe,
              tx = tx,
              coords = NULL,
              distance = 50, nbins = 200, cl = 1,
              out_dir = "denoist_results")
```

With a count matrix and coordinates:

```         
library(DenoIST)

res <- denoist(mat = mat,
               tx = tx,
               coords = coords,
               distance = 50, nbins = 200, cl = 1,
               out_dir = "denoist_results")
```

## R Vignette

Check out the vignette to get started:

```
library(DenoIST)
browseVignettes("DenoIST")
```
