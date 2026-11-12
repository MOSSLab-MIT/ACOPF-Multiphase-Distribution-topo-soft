"""scripts/slurm_runs/grid_config.py: `train_set` -> TRAIN_H5 / TRAIN_TAG bash variables."""

import importlib.util
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

SLURM_DIR = Path(__file__).resolve().parents[1] / "scripts" / "slurm_runs"
GRIDS = ("13", "123", "240", "906")


@pytest.fixture()
def module():
    spec = importlib.util.spec_from_file_location("grid_config", SLURM_DIR / "grid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(module, monkeypatch, capsys, grid):
    monkeypatch.setattr("sys.argv", ["grid_config.py", grid])
    module.main()
    assignments = {}
    for line in capsys.readouterr().out.splitlines():
        name, _, value = line.partition("=")
        assignments[name] = value
    return assignments


def _write_config(tmp_path, monkeypatch, module, **overrides):
    cfg = yaml.safe_load((SLURM_DIR / "configs" / "13.yaml").read_text())
    cfg.update(overrides)
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(module, "CONFIG_DIR", tmp_path)
    return cfg


@pytest.mark.parametrize("grid", GRIDS)
def test_checked_in_configs_use_the_default_training_set(module, monkeypatch, capsys, grid):
    cfg = yaml.safe_load((SLURM_DIR / "configs" / f"{grid}.yaml").read_text())
    out = _run(module, monkeypatch, capsys, grid)
    case_dir = cfg["case_dir"]
    expected = f"datagen_topo/datasets/{case_dir}/{case_dir}_training.h5"
    assert shlex.split(out["TRAIN_H5"]) == [expected]
    assert shlex.split(out["TRAIN_TAG"]) == [""]


def test_suffixed_training_set(module, monkeypatch, capsys, tmp_path):
    cfg = _write_config(tmp_path, monkeypatch, module, train_set="training_060_005_035")
    out = _run(module, monkeypatch, capsys, "x")
    case_dir = cfg["case_dir"]
    expected = f"datagen_topo/datasets/{case_dir}/{case_dir}_training_060_005_035.h5"
    assert shlex.split(out["TRAIN_H5"]) == [expected]
    assert shlex.split(out["TRAIN_TAG"]) == ["_060_005_035"]


def test_missing_train_set_is_rejected(module, monkeypatch, capsys, tmp_path):
    cfg = yaml.safe_load((SLURM_DIR / "configs" / "13.yaml").read_text())
    del cfg["train_set"]
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["grid_config.py", "x"])
    with pytest.raises(SystemExit, match="missing keys .*train_set"):
        module.main()


@pytest.mark.parametrize(
    "bad",
    ["", "training_", "train", "_training", "test_feasible", "training/x", "training x", None, 1],
)
def test_invalid_train_set_is_rejected(module, monkeypatch, capsys, tmp_path, bad):
    _write_config(tmp_path, monkeypatch, module, train_set=bad)
    monkeypatch.setattr("sys.argv", ["grid_config.py", "x"])
    with pytest.raises(SystemExit, match="train_set must be"):
        module.main()


def test_empty_tag_survives_bash_eval_under_nounset(module, monkeypatch, capsys):
    """The default tag must be defined-but-empty: method scripts run under `set -u`."""
    monkeypatch.setattr("sys.argv", ["grid_config.py", "13"])
    module.main()
    script = capsys.readouterr().out + '\nset -u\nprintf "[%s]" "$TRAIN_TAG"\n'
    assert shutil.which("bash") is not None
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout == "[]"
