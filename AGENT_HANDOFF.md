# Agent Handoff Notes

This repo is primarily an R package for DenoIST, with an experimental Python
prototype added under `denoistpy/`. These notes are intended to help future
agents or maintainers pick up the Python port without rediscovering the current
design decisions.

## Current Python Architecture

- `denoistpy.core.denoist()` is the main Python pipeline.
- `denoistpy.io` contains container ingress helpers:
  - `from_anndata()`
  - `from_spatialdata()`
  - `from_proseg_spatialdata()`
- `denoistpy.offsets` handles transcript filtering, hex-bin background
  estimation, sparse radius-neighbor adjacency, and sparse local offset
  aggregation.
- `denoistpy.mixture` contains NumPy and PyTorch Poisson mixture EM solvers.
- `denoistpy.export` contains output helpers:
  - `result_to_anndata()`
  - `add_result_table_to_spatialdata()`
  - `write_result_h5ad()`
  - `write_result_anndata_zarr()`
  - `write_result_spatialdata_zarr()`
- `denoistpy.report` contains report helpers:
  - `summarize_denoist_result()`
  - `write_report_csv()`
- `denoistpy.cli` contains the console entry point:
  - `denoistpy run-proseg`
- `tests_py/` contains lightweight Python tests.

## Orientation Convention

The Python internals currently follow the R package convention:

```text
genes x cells
```

This is different from `AnnData.X`, which is conventionally:

```text
cells x genes
```

Boundary helpers transpose as needed:

- `from_anndata()` converts `AnnData.X` from cells x genes to genes x cells.
- `result_to_anndata()` converts DenoIST outputs back to cells x genes.

This convention is deliberate for now because it makes R parity checks easier.
The algorithm itself does not require this orientation, so an AnnData-native
internal refactor is possible later, but should be done with tests around sparse
matrix orientation and neighbor aggregation.

## Implemented

- Sparse-first count handling with SciPy sparse matrices.
- Optional `SpatialData`/`AnnData` input helpers, including a Proseg-specific
  helper for the common `table` plus `transcripts` layout.
- Transcript QV filtering.
- Occupied hexagonal transcript binning for background estimation.
- Gaussian mixture background-bin selection with a fallback quantile path.
- Sparse radius-neighbor adjacency using `scipy.spatial.cKDTree`.
- Sparse local offset aggregation via `counts @ adjacency`.
- Local neighbourhood offsets include each cell's own counts once by default.
  `include_self_twice=False` is the cleaner default. Set
  `include_self_twice=True` only for compatibility checks against the current R
  fast path, which appears to include self in the adjacency and then add the
  original count matrix again.
- Background is kept as a per-gene vector until EM batching to avoid expanding
  a dense genes x cells offset matrix.
- NumPy per-cell Poisson mixture solver.
- PyTorch batched Poisson mixture solver for dense gene-by-cell chunks.
- PyTorch backend tests cover both `device="cpu"` and `device="cuda"` when
  CUDA is available.
- Sparse adjusted count output.
- Optional dense memberships and posterior storage.
- Zero-total cells are retained in outputs but skipped in the Poisson mixture
  model. Their memberships/posteriors are set to 1 if requested, adjusted
  counts remain zero, and params carry `status == "zero_count"`.
- `denoist(..., background_only=True)` skips neighbor aggregation and Poisson
  mixture fitting. It estimates per-gene ambient background from transcript hex
  bins and applies `max(count - background_gene, 0)` to sparse nonzero counts.
  Params carry `status == "background_only"` for nonzero cells.
- AnnData export with raw/corrected count layers, model params in `obs`, and
  run metadata in `uns`. Export can preserve a template AnnData table's `obs`,
  `var`, `obsm`, and optionally `uns`.
- SpatialData-like table insertion by copying the original object and adding a
  DenoIST AnnData table.
- Writer helpers for H5AD, AnnData Zarr, and SpatialData Zarr outputs.
- Report helpers summarize removed count mass overall, per gene, per cell, top
  removed genes/cells, model status counts, and background-estimation
  diagnostics. AnnData exports annotate `obs`, `var`, and `uns["denoist"]` with
  removed-count metrics.
- CLI support currently covers Proseg-style SpatialData Zarr input via
  `denoistpy run-proseg`, with output formats `spatialdata-zarr`, `h5ad`, and
  `anndata-zarr`, plus optional CSV reports.
- Optional R/Python parity scaffolding lives under `tests_py/parity/`. It uses
  conda-managed R via `tests_py/parity/environment-r.yml`, generates portable
  R references with `generate_r_reference.R`, and compares rough totals in
  `tests_py/test_r_parity.py`.

## Known Gaps / Non-Parity With R

- Python hex binning is a native axial-coordinate implementation. It follows
  the same conceptual role and area scaling as R `hexbin`, but is not yet
  proven bitwise-identical to R's `hexbin(..., IDs = TRUE)`.
- No regression tests against the R `.rds` fixtures yet.
- `from_spatialdata()` is intentionally minimal and duck-typed. It assumes one
  table-like AnnData object plus a points element convertible to pandas.
- `from_proseg_spatialdata()` handles the observed Proseg layout:
  `sdata.tables["table"]` as cells x genes AnnData and
  `sdata.points["transcripts"]` with `x`, `y`, `observed_x`, `observed_y`,
  `gene`, and optional `background`. It supports adjusted vs observed
  transcript positions and optional Proseg background filtering.
- Proseg support is based on one representative ROI structure and should still
  be tested against more datasets.
- Sparse neighbor aggregation is CPU-only. PyTorch acceleration currently
  applies to the EM stage, not sparse matrix multiplication.
- No CuPy/RAPIDS path yet.
- H5AD/Zarr writers exist, but they still materialize an in-memory AnnData
  export object before writing. There is no streaming writer for huge posterior
  or membership outputs yet.
- No parity validation for the R fast-path behavior where self counts may be
  included in the neighbor offset and then added again. Python now defaults to
  `include_self_twice=False`; use `True` only for R compatibility checks.

## Test Environment

Typical Python test commands:

```powershell
python --version
python -m pytest tests_py
```

If dependencies need reinstalling:

```powershell
python -m pip install --editable ".[spatialdata,test]"
```

The PyTorch backend supports `device="auto"`, which resolves to CUDA, then MPS,
then CPU. Install a platform-appropriate PyTorch build before running GPU tests.

For R parity reference generation:

```powershell
conda env create -f tests_py/parity/environment-r.yml
conda run -n denoistpy_r Rscript tests_py/parity/generate_r_reference.R
python -m pytest tests_py/test_r_parity.py
```

On Windows, `bioconductor-sparsematrixstats` was not available through conda
for this environment. The parity script therefore defines a local `rowSums2`
shim before sourcing the R implementation; this is sufficient for the current
small matrix fixture path used by the reference generator.

Python bytecode, pytest caches, writer smoke-test outputs, and generated parity
CSVs should stay untracked.

## Sensible Next Steps

1. Add R/Python parity fixtures:
   - run R DenoIST on the small test data
   - run Python on the same data
   - compare offsets, memberships, adjusted counts, and params
2. Quantify the sensitivity of outputs to `include_self_twice=True` versus the
   cleaner default `False`.
3. Add real Proseg fixture/round-trip tests, including Dask-backed transcript
   points with multiple partitions.
4. Add streaming/chunked H5AD/Zarr writing for large outputs.
5. Add optional CuPy sparse matrix multiplication for `counts @ adjacency`.
6. Add a benchmark script for Xenium/CosMx-style matrix sizes.

## Important Files

- `R/denoist.R`: R wrapper and reference orchestration.
- `R/neighbour_offset.R`: R local/background offset implementation.
- `R/pmm_model.R`: R Poisson mixture model.
- `denoistpy/core.py`: Python orchestration.
- `denoistpy/offsets.py`: Python background and neighbor offset logic.
- `denoistpy/mixture.py`: Python NumPy/PyTorch EM solvers.
- `denoistpy/export.py`: Python AnnData/SpatialData output helpers.
- `denoistpy/report.py`: Python report summaries and CSV writer.
- `denoistpy/cli.py`: Python CLI parser and `run-proseg` command.
- `tests_py/test_denoistpy.py`: Python smoke and output-format tests.
- `tests_py/test_r_parity.py`: Optional R/Python parity checks; skipped until
  R references are generated.
- `tests_py/parity/environment-r.yml`: Conda R environment for reference
  generation.
- `tests_py/parity/generate_r_reference.R`: R reference-output generator.
