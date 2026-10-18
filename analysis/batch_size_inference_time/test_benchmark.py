"""Tests for batch-size inference-time benchmark (penalty_fs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.batch_size_inference_time.benchmark import (
    BENCHMARK_NAME,
    GRID_PRESETS,
    align_test_samples,
    pending_batch_sizes,
    regroup_batches,
    resolve_grid,
    resolve_power_flow_solver,
    resolve_run_checkpoint,
    run_benchmark,
    write_results,
    _result_status,
    _validate_resume_compatible,
)
from core.grids.power_flow import PowerFlowSolver
from analysis.batch_size_inference_time.plot import load_result_files, plot_batch_size_curves

IEEE13_CKPT_ROOT = (
    _ROOT
    / "results"
    / "penalty_fs"
    / "IEEE13Nodeckt_penalty_fs_rho1e5"
    / "seed_1"
    / "models"
)
_TRAIN_H5 = _ROOT / GRID_PRESETS["13"]["train_h5"]
_TEST_H5 = _ROOT / GRID_PRESETS["13"]["test_h5"]

_REQUIRED_TOP_KEYS = {
    "benchmark",
    "method",
    "case_suffix",
    "grid",
    "variant",
    "warmup",
    "repeats",
    "test_samples",
    "device",
    "timed_scope",
    "excludes",
    "by_batch_size",
}
_REQUIRED_BATCH_KEYS = {
    "pipeline_time_per_batch_mean_s",
    "pipeline_time_per_batch_std_s",
    "raw_time_per_batch_mean_s",
    "fs_time_per_batch_mean_s",
    "inverse_time_per_batch_mean_s",
    "pipeline_time_per_sample_mean_s",
    "num_batches",
    "batch_size",
}


def _has_ieee13_assets() -> bool:
    return (
        _TRAIN_H5.exists()
        and _TEST_H5.exists()
        and IEEE13_CKPT_ROOT.exists()
        and any(IEEE13_CKPT_ROOT.glob("checkpoint_epoch_*"))
    )


pytestmark = pytest.mark.skipif(
    not _has_ieee13_assets(),
    reason="IEEE13 data or penalty_fs_rho1e5 seed_1 checkpoint missing",
)


def test_align_test_samples():
    assert align_test_samples(256, [1, 2, 4, 8, 16, 32, 64]) == 256
    assert align_test_samples(250, [1, 2, 4, 8, 16, 32, 64]) == 192
    with pytest.raises(ValueError):
        align_test_samples(32, [64])


def test_resolve_power_flow_solver_grid_overrides():
    cfg_fixed = {"power_flow_solver": PowerFlowSolver.FIXED_POINT.value}
    cfg_newton = {"power_flow_solver": PowerFlowSolver.NEWTON_GMRES.value}
    assert resolve_power_flow_solver("240", cfg_fixed) == PowerFlowSolver.NEWTON_GMRES.value
    assert resolve_power_flow_solver("240", cfg_newton) == PowerFlowSolver.NEWTON_GMRES.value
    assert (
        resolve_power_flow_solver("13", cfg_fixed) == PowerFlowSolver.FIXED_POINT.value
    )
    assert resolve_power_flow_solver("13", {}) == PowerFlowSolver.FIXED_POINT.value
    assert (
        resolve_power_flow_solver("906", cfg_newton) == PowerFlowSolver.NEWTON_GMRES.value
    )


def test_regroup_batches_drops_remainder():
    import numpy as np

    inputs = np.arange(10)[:, None]
    targets = np.arange(10)[:, None] * 2
    batches = regroup_batches(inputs, targets, batch_size=4)
    assert len(batches) == 2
    assert batches[0][0].shape[0] == 4
    assert batches[1][0].shape[0] == 4


def test_pending_batch_sizes_and_status():
    done = {"1": {}, "2": {}}
    assert pending_batch_sizes([1, 2, 4, 8], done) == [4, 8]
    assert _result_status([1, 2, 4, 8], done) == "partial"
    assert _result_status([1, 2, 4, 8], done, failed_batch_size=4) == "failed"
    assert _result_status([1, 2], {"1": {}, "2": {}}) == "complete"


def test_validate_resume_compatible_rejects_mismatch():
    existing = {
        "grid": "13",
        "case_suffix": "penalty_fs_rho1e5",
        "warmup": 2,
        "repeats": 3,
        "test_samples_requested": 256,
        "test_samples": 256,
        "batch_sizes": [1, 2],
    }
    _validate_resume_compatible(
        existing,
        out_path=Path("results_13.json"),
        grid="13",
        case_suffix="penalty_fs_rho1e5",
        batch_sizes=[1, 2],
        warmup=2,
        repeats=3,
        test_samples_requested=256,
        aligned_test_samples=256,
    )
    with pytest.raises(ValueError, match="warmup"):
        _validate_resume_compatible(
            existing,
            out_path=Path("results_13.json"),
            grid="13",
            case_suffix="penalty_fs_rho1e5",
            batch_sizes=[1, 2],
            warmup=99,
            repeats=3,
            test_samples_requested=256,
            aligned_test_samples=256,
        )


def test_resolve_run_checkpoint_from_results_root():
    ckpt = resolve_run_checkpoint(
        checkpoint_path=None,
        results_root=str(_ROOT / "results" / "penalty_fs"),
        case_dir="IEEE13Nodeckt",
        case_suffix="penalty_fs_rho1e5",
        seed=1,
    )
    assert ckpt.name.startswith("checkpoint_epoch_")
    assert (ckpt / "config.json").exists()
    # Prefer models/last/ (final train save), not an intermediate epoch under models/.
    assert ckpt.parent.name == "last"
    assert ckpt.name == "checkpoint_epoch_10"


def test_smoke_benchmark_and_plot(tmp_path: Path):
    """Tiny end-to-end: B in {1,2}, warmup=1, repeats=1, N=4."""
    out_json = tmp_path / "results_13.json"
    ckpt = resolve_run_checkpoint(
        checkpoint_path=None,
        results_root=str(_ROOT / "results" / "penalty_fs"),
        case_dir=resolve_grid("13")["case_dir"],
        case_suffix="penalty_fs_rho1e5",
        seed=1,
    )
    payload = run_benchmark(
        grid="13",
        checkpoint_path=ckpt,
        batch_sizes=(1, 2),
        test_samples=4,
        warmup=1,
        repeats=1,
        seed=1,
        case_suffix="penalty_fs_rho1e5",
        out_path=out_json,
        resume=True,
    )
    assert payload["benchmark"] == BENCHMARK_NAME
    assert set(payload) >= _REQUIRED_TOP_KEYS
    assert payload["status"] == "complete"
    assert payload["test_samples"] == 4
    assert payload["timed_scope"] == "raw_nn_project_pf_plus_post_process_slfs"
    assert "topology_inverse" in payload["excludes"]

    for b in ("1", "2"):
        block = payload["by_batch_size"][b]
        assert set(block) >= _REQUIRED_BATCH_KEYS
        assert block["batch_size"] == int(b)
        assert block["pipeline_time_per_batch_mean_s"] > 0.0
        assert block["num_batches"] == 4 // int(b)

    out_json = tmp_path / "results_13.json"
    write_results(payload, out_json)
    loaded = json.loads(out_json.read_text())
    assert loaded["grid"] == "13"

    payloads = load_result_files(tmp_path)
    assert "13" in payloads
    out_pdf = tmp_path / "batch_size_inference_time.pdf"
    plot_batch_size_curves(payloads, out_pdf)
    assert out_pdf.exists() and out_pdf.stat().st_size > 0
