from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from .core import denoist
from .export import write_result_anndata_zarr, write_result_h5ad, write_result_spatialdata_zarr
from .io import from_proseg_spatialdata
from .report import summarize_denoist_result, write_report_csv


def _parse_n_inits(value: str) -> int | np.ndarray:
    if "," not in value:
        return int(value)
    vals = np.asarray([float(v.strip()) for v in value.split(",") if v.strip()], dtype=np.float32)
    if vals.size == 0:
        raise argparse.ArgumentTypeError("n-inits vector cannot be empty.")
    return vals


def _optional_str(value: str | None) -> str | None:
    if value in {None, "", "none", "None", "NULL", "null"}:
        return None
    return value


def _read_spatialdata(path: Path):
    try:
        import spatialdata as sd
    except ImportError as exc:
        raise SystemExit(
            "The run-proseg command requires spatialdata. Install with `pip install denoistpy[spatialdata]`."
        ) from exc
    return sd.read_zarr(path)


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--distance", type=float, default=50.0, help="Neighbour/background distance radius.")
    parser.add_argument("--nbins", type=int, default=200, help="Number of x-axis hex bins for background estimation.")
    parser.add_argument("--posterior-cutoff", type=float, default=0.6, help="Posterior cutoff for mixture memberships.")
    parser.add_argument("--n-inits", type=_parse_n_inits, default=10, help="Number of pi initializations or comma vector.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=1e-6, help="EM convergence tolerance.")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy", help="Poisson mixture backend.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cpu, or mps.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Cells per dense EM batch.")
    parser.add_argument("--background-only", action="store_true", help="Skip mixture model and subtract ambient background only.")
    parser.add_argument("--include-self-twice", action="store_true", help="Compatibility mode for the R fast-path self-count behavior.")
    parser.add_argument("--store-memberships", action="store_true", help="Store dense genes x cells membership matrix.")
    parser.add_argument("--store-posterior", action="store_true", help="Store dense genes x cells posterior matrix.")
    parser.add_argument(
        "--progress",
        choices=["none", "text", "tqdm", "auto"],
        default="text",
        help="Progress reporting style. CLI defaults to lightweight text.",
    )


def _add_proseg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Input Proseg/SpatialData Zarr store.")
    parser.add_argument("output", type=Path, help="Output path.")
    parser.add_argument(
        "--output-format",
        choices=["spatialdata-zarr", "h5ad", "anndata-zarr"],
        default="spatialdata-zarr",
        help="Output container format.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output path.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Optional directory for CSV QC report tables.")
    parser.add_argument("--top-n", type=int, default=25, help="Top N genes/cells to include in report tables.")
    parser.add_argument("--table-key", default="table", help="SpatialData table key containing cells x genes counts.")
    parser.add_argument("--points-key", default="transcripts", help="SpatialData points key containing transcripts.")
    parser.add_argument("--output-table-key", default="denoist", help="Output table key for SpatialData Zarr output.")
    parser.add_argument("--layer", default=None, help="AnnData layer to use for raw counts; defaults to X.")
    parser.add_argument("--spatial-key", default="spatial", help="AnnData obsm key for cell coordinates.")
    parser.add_argument(
        "--cell-coordinate-source",
        choices=["obsm", "obs"],
        default="obsm",
        help="Source of cell coordinates.",
    )
    parser.add_argument("--cell-x-col", default=None, help="obs column for cell x coordinate when using --cell-coordinate-source obs.")
    parser.add_argument("--cell-y-col", default=None, help="obs column for cell y coordinate when using --cell-coordinate-source obs.")
    parser.add_argument("--gene-names-col", default=None, help="Optional var column to use as gene names.")
    parser.add_argument(
        "--transcript-position",
        choices=["adjusted", "observed"],
        default="adjusted",
        help="Use Proseg adjusted x/y or observed_x/observed_y transcript positions.",
    )
    parser.add_argument("--x-col", default=None, help="Explicit transcript x column override.")
    parser.add_argument("--y-col", default=None, help="Explicit transcript y column override.")
    parser.add_argument("--gene-col", default="gene", help="Transcript gene column.")
    parser.add_argument("--qv-col", default=None, help="Transcript QV column; Proseg usually has none.")
    parser.add_argument("--qv-threshold", type=float, default=20.0, help="QV filter threshold if --qv-col is present.")
    parser.add_argument(
        "--background-filter",
        choices=["none", "exclude", "only"],
        default="none",
        help="How to use Proseg transcript background flag.",
    )
    parser.add_argument("--background-col", default="background", help="Proseg background flag column.")
    parser.add_argument("--copy-uns-from-template", action="store_true", help="Copy source table uns into exported AnnData.")
    _add_common_run_args(parser)


def run_proseg(args: argparse.Namespace) -> int:
    sdata = _read_spatialdata(args.input)
    inp = from_proseg_spatialdata(
        sdata,
        table_key=args.table_key,
        points_key=args.points_key,
        layer=_optional_str(args.layer),
        transcript_position=args.transcript_position,
        x_col=_optional_str(args.x_col),
        y_col=_optional_str(args.y_col),
        gene_col=args.gene_col,
        background_col=args.background_col,
        background_filter=None if args.background_filter == "none" else args.background_filter,
        spatial_key=args.spatial_key,
        cell_coordinate_source=args.cell_coordinate_source,
        cell_x_col=_optional_str(args.cell_x_col),
        cell_y_col=_optional_str(args.cell_y_col),
        gene_names_col=_optional_str(args.gene_names_col),
    )
    result = denoist(
        inp,
        x_col="x",
        y_col="y",
        gene_col=args.gene_col,
        qv_col=_optional_str(args.qv_col),
        qv_threshold=args.qv_threshold,
        distance=args.distance,
        nbins=args.nbins,
        posterior_cutoff=args.posterior_cutoff,
        n_inits=args.n_inits,
        max_iter=args.max_iter,
        tol=args.tol,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        store_memberships=args.store_memberships,
        store_posterior=args.store_posterior,
        background_only=args.background_only,
        include_self_twice=args.include_self_twice,
        progress=args.progress,
    )

    common_export = {
        "template_adata": inp.source_table,
        "copy_uns_from_template": args.copy_uns_from_template,
    }
    if args.output_format == "spatialdata-zarr":
        write_result_spatialdata_zarr(
            sdata,
            result,
            inp,
            args.output,
            table_key=args.output_table_key,
            overwrite=args.overwrite,
            **common_export,
        )
    elif args.output_format == "h5ad":
        write_result_h5ad(result, inp, args.output, overwrite=args.overwrite, **common_export)
    elif args.output_format == "anndata-zarr":
        write_result_anndata_zarr(result, inp, args.output, overwrite=args.overwrite, **common_export)
    else:
        raise AssertionError(f"Unhandled output format: {args.output_format}")

    if args.report_dir is not None:
        report = summarize_denoist_result(result, inp, top_n=args.top_n)
        write_report_csv(report, args.report_dir, overwrite=args.overwrite)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="denoistpy", description="Sparse-first Python DenoIST tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-proseg", help="Run DenoIST on Proseg-style SpatialData Zarr output.")
    _add_proseg_args(run_parser)
    run_parser.set_defaults(func=run_proseg)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
