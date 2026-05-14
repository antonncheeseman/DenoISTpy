import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from denoistpy import (
    add_result_table_to_spatialdata,
    denoist,
    from_proseg_spatialdata,
    local_offset_distance_with_background,
    result_to_anndata,
    summarize_denoist_result,
    write_result_anndata_zarr,
    write_result_h5ad,
    write_report_csv,
    write_result_spatialdata_zarr,
)
from denoistpy.cli import build_parser, main
from denoistpy.offsets import _hex_gene_bin_matrix
from denoistpy.mixture import solve_poisson_mixture_numpy


def make_output_dir(name):
    path = Path(name)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def toy_inputs():
    counts = sparse.csr_matrix(
        np.array(
            [
                [3, 0, 1],
                [0, 5, 1],
                [2, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    coords = np.array([[0, 0], [10, 0], [100, 100]], dtype=float)
    transcripts = pd.DataFrame(
        {
            "x": [0, 1, 10, 11, 100, 101, 5, 6],
            "y": [0, 1, 0, 1, 100, 101, 5, 6],
            "gene": ["g1", "g1", "g2", "g2", "g3", "g3", "g1", "g2"],
            "qv": [30, 30, 30, 30, 30, 30, 10, 10],
        }
    )
    return counts, coords, transcripts, np.array(["g1", "g2", "g3"])


def test_numpy_mixture_shapes():
    out = solve_poisson_mixture_numpy(
        np.array([3, 5, 2, 8, 6], dtype=np.float32),
        np.array([1, 0, 1, 0, 1], dtype=np.float32),
        n_inits=np.array([0.1, 0.2]),
    )
    assert out["memberships"].shape == (5,)
    assert np.all(out["memberships"][[1, 3]] == 1)


def test_cli_help_and_run_proseg_parser(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "run-proseg" in capsys.readouterr().out

    parser = build_parser()
    args = parser.parse_args(
        [
            "run-proseg",
            "input.zarr",
            "output.zarr",
            "--output-format",
            "h5ad",
            "--transcript-position",
            "observed",
            "--background-filter",
            "exclude",
            "--n-inits",
            "0.1,0.2",
            "--background-only",
        ]
    )
    assert args.output_format == "h5ad"
    assert args.transcript_position == "observed"
    assert args.background_filter == "exclude"
    assert np.allclose(args.n_inits, np.array([0.1, 0.2], dtype=np.float32))
    assert args.background_only is True


def test_numpy_mixture_skips_zero_count_cell():
    out = solve_poisson_mixture_numpy(
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        n_inits=np.array([0.1, 0.2]),
    )
    assert out["status"] == "zero_count"
    assert np.all(out["memberships"] == 1)
    assert np.all(out["posterior"] == 1)
    assert np.isnan(out["lambda1"])


def test_offsets_shape():
    counts, coords, transcripts, genes = toy_inputs()
    offsets = local_offset_distance_with_background(
        counts,
        coords,
        transcripts,
        gene_names=genes,
        distance=20,
        nbins=10,
    )
    assert offsets.shape == counts.shape


def test_hex_binning_groups_nearby_transcripts():
    tx = pd.DataFrame(
        {
            "x": [0.0, 0.1, 10.0],
            "y": [0.0, 0.1, 10.0],
            "gene": ["g1", "g1", "g2"],
        }
    )
    mat, area = _hex_gene_bin_matrix(
        tx,
        {"g1": 0, "g2": 1},
        x_col="x",
        y_col="y",
        gene_col="gene",
        nbins=2,
    )
    assert mat.shape[0] == 2
    assert mat.shape[1] == 2
    assert area > 0
    assert mat[0, :].max() == 2


def test_denoist_numpy_result_shape():
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
        batch_size=2,
    )
    assert result.adjusted_counts.shape == counts.shape
    assert result.memberships.shape == counts.shape
    assert len(result.params) == counts.shape[1]
    assert result.metadata["include_self_twice"] is False


def test_denoist_keeps_zero_count_cells_but_skips_fit():
    counts, coords, transcripts, genes = toy_inputs()
    counts = counts.tolil()
    counts[:, 1] = 0
    counts = counts.tocsr()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )
    zero_cell = result.params.loc[result.params["cell_index"] == 1].iloc[0]
    assert zero_cell["status"] == "zero_count"
    assert zero_cell["n_iter"] == 0
    assert np.isnan(zero_cell["lambda1"])
    assert np.all(result.memberships[:, 1] == 1)
    assert result.adjusted_counts[:, 1].nnz == 0


def test_denoist_background_only_subtracts_ambient_without_em():
    counts = sparse.csr_matrix(
        np.array(
            [
                [3, 0],
                [1, 2],
            ],
            dtype=np.float32,
        )
    )
    coords = np.array([[0, 0], [10, 0]], dtype=float)
    transcripts = pd.DataFrame({"x": [0.0], "y": [0.0], "gene": ["g1"]})
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=np.array(["g1", "g2"]),
        distance=20,
        nbins=10,
        background_only=True,
    )
    expected = np.array(
        [
            [2, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(result.adjusted_counts.toarray(), expected)
    assert set(result.params["status"]) == {"background_only"}
    assert np.all(result.params["n_iter"] == 0)
    assert result.metadata["mode"] == "background_only"


def test_denoist_torch_backend_cpu_result_shape():
    pytest.importorskip("torch")
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
        backend="torch",
        device="cpu",
        batch_size=2,
    )
    assert result.adjusted_counts.shape == counts.shape
    assert result.memberships.shape == counts.shape
    assert result.metadata["backend"] == "torch"


def test_denoist_torch_backend_cuda_result_shape():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
        backend="torch",
        device="cuda",
        batch_size=2,
    )
    assert result.adjusted_counts.shape == counts.shape
    assert result.memberships.shape == counts.shape
    assert result.metadata["backend"] == "torch"


def test_denoist_can_skip_dense_memberships():
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
        store_memberships=False,
    )
    assert result.adjusted_counts.shape == counts.shape
    assert result.memberships is None


def test_result_to_anndata_layers():
    pytest.importorskip("anndata")
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )
    adata = result_to_anndata(result, counts)
    assert adata.shape == (counts.shape[1], counts.shape[0])
    assert "raw_counts" in adata.layers
    assert "denoist_corrected" in adata.layers
    assert "denoist_lambda1" in adata.obs
    assert "denoist_removed_counts" in adata.obs
    assert "denoist_removed_counts" in adata.var
    assert "denoist" in adata.uns
    assert "summary" in adata.uns["denoist"]


def test_add_result_table_to_spatialdata_like_object():
    pytest.importorskip("anndata")

    class FakeSpatialData:
        def __init__(self):
            self.tables = {}

    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )
    original = FakeSpatialData()
    updated = add_result_table_to_spatialdata(original, result, counts, table_key="denoist")
    assert "denoist" not in original.tables
    assert "denoist" in updated.tables
    assert updated.tables["denoist"].layers["denoist_corrected"].shape == (
        counts.shape[1],
        counts.shape[0],
    )


def test_from_proseg_spatialdata_uses_observed_coordinates_and_filters_background():
    ad = pytest.importorskip("anndata")

    class FakeSpatialData:
        def __init__(self, table, points):
            self.tables = {"table": table}
            self.points = {"transcripts": points}

    table = ad.AnnData(
        X=sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "cell": [0, 1],
                "centroid_x": [10.0, 20.0],
                "centroid_y": [30.0, 40.0],
                "region": ["cell_boundaries", "cell_boundaries"],
            },
            index=["cell0", "cell1"],
        ),
        var=pd.DataFrame({"gene": ["g1", "g2"], "total_count": [1, 2]}, index=["g1", "g2"]),
    )
    table.obsm["spatial"] = np.array([[10.0, 30.0], [20.0, 40.0]], dtype=np.float32)
    table.uns["proseg_run"] = {"version": "test"}
    points = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "observed_x": [101.0, 102.0, 103.0],
            "observed_y": [104.0, 105.0, 106.0],
            "gene": ["g1", "g2", "g1"],
            "background": [False, True, False],
            "assignment": [0, 1, 0],
        }
    )

    inp = from_proseg_spatialdata(
        FakeSpatialData(table, points),
        transcript_position="observed",
        background_filter="exclude",
    )

    assert inp.counts.shape == (2, 2)
    assert np.array_equal(inp.gene_names, np.array(["g1", "g2"]))
    assert np.array_equal(inp.cell_names, np.array(["cell0", "cell1"]))
    assert list(inp.transcripts.columns) == ["x", "y", "gene", "background"]
    assert inp.transcripts["x"].tolist() == [101.0, 103.0]
    assert inp.transcripts["y"].tolist() == [104.0, 106.0]
    assert inp.source_table is table


def test_result_to_anndata_preserves_template_metadata():
    ad = pytest.importorskip("anndata")
    counts, coords, transcripts, genes = toy_inputs()
    template = ad.AnnData(
        X=counts.T.tocsr(),
        obs=pd.DataFrame({"region": ["a", "b", "c"]}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame({"gene": genes, "total_count": [4, 6, 3]}, index=genes),
    )
    template.obsm["spatial"] = coords
    template.uns["proseg_run"] = {"version": "test"}
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        cell_names=np.array(["c0", "c1", "c2"]),
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )

    adata = result_to_anndata(
        result,
        counts,
        template_adata=template,
        copy_uns_from_template=True,
    )

    assert "region" in adata.obs
    assert "denoist_lambda1" in adata.obs
    assert "total_count" in adata.var
    assert "spatial" in adata.obsm
    assert "proseg_run" in adata.uns


def test_write_result_h5ad_and_zarr():
    ad = pytest.importorskip("anndata")
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )

    out = make_output_dir("writer-test-output")
    try:
        h5ad_path = write_result_h5ad(result, counts, out / "result.h5ad")
        zarr_path = write_result_anndata_zarr(result, counts, out / "result.zarr")

        assert h5ad_path.exists()
        assert zarr_path.exists()
        loaded = ad.read_h5ad(h5ad_path)
        assert loaded.shape == (counts.shape[1], counts.shape[0])
        assert "denoist_corrected" in loaded.layers
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_summarize_and_write_report_csv():
    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )
    report = summarize_denoist_result(result, counts, top_n=2)
    assert set(report) == {
        "summary",
        "per_gene",
        "per_cell",
        "top_removed_genes",
        "top_removed_cells",
        "model_status",
    }
    assert len(report["top_removed_genes"]) == 2
    assert "removed_counts_total" in set(report["summary"]["metric"])
    assert "background_n_transcripts_used" in set(report["summary"]["metric"])

    out = make_output_dir("report-test-output")
    try:
        report_path = write_report_csv(report, out, overwrite=True)
        assert (report_path / "summary.csv").exists()
        assert (report_path / "per_gene.csv").exists()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_write_result_spatialdata_zarr_uses_write_method():
    pytest.importorskip("anndata")

    class FakeSpatialData:
        def __init__(self):
            self.tables = {}
            self.write_args = None

        def write(self, file_path, overwrite=False, consolidate_metadata=True, update_sdata_path=True):
            self.write_args = {
                "file_path": file_path,
                "overwrite": overwrite,
                "consolidate_metadata": consolidate_metadata,
                "update_sdata_path": update_sdata_path,
                "tables": set(self.tables),
            }
            file_path.mkdir(parents=True, exist_ok=True)

        def __deepcopy__(self, memo):
            copied = FakeSpatialData()
            copied.tables = dict(self.tables)
            return copied

    counts, coords, transcripts, genes = toy_inputs()
    result = denoist(
        counts,
        transcripts,
        coords,
        gene_names=genes,
        distance=20,
        nbins=10,
        n_inits=np.array([0.1, 0.2]),
    )
    original = FakeSpatialData()
    out = make_output_dir("writer-sdata-test-output")
    try:
        output = out / "sdata.zarr"
        out_path = write_result_spatialdata_zarr(
            original,
            result,
            counts,
            output,
            overwrite=True,
            copy_sdata=False,
        )

        assert out_path == output
        assert output.exists()
        assert original.write_args["overwrite"] is True
        assert "denoist" in original.write_args["tables"]
    finally:
        shutil.rmtree(out, ignore_errors=True)
