"""HDF5 price provenance (metadata/{sub,der}_cost_{lin,quad}) must match network_config.json."""

import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.grids.data import _COST_METADATA_KEYS, _check_cost_metadata

NET = SimpleNamespace(
    sub_lin=np.float64(100.0), sub_quad=np.float64(1.0),
    der_lin=np.full(4, 5.0), der_quad=np.full(4, 0.1),
)
MATCH = {"sub_cost_lin": 100.0, "sub_cost_quad": 1.0,
         "der_cost_lin": np.full(4, 5.0), "der_cost_quad": np.full(4, 0.1)}


def _write(path, meta):
    with h5py.File(path, "w") as f:
        f["metadata/n_net"] = 3
        for k, v in meta.items():
            f[f"metadata/{k}"] = v
    return str(path)


def test_matching_price_accepted(tmp_path):
    _check_cost_metadata(_write(tmp_path / "a.h5", MATCH), NET)


def test_legacy_file_without_price_accepted(tmp_path):
    _check_cost_metadata(_write(tmp_path / "a.h5", {}), NET)
    with h5py.File(tmp_path / "b.h5", "w") as f:
        f["input/pd"] = np.zeros((2, 3))
    _check_cost_metadata(str(tmp_path / "b.h5"), NET)


@pytest.mark.parametrize("key", _COST_METADATA_KEYS)
def test_any_mismatch_rejected(tmp_path, key):
    meta = dict(MATCH)
    bad = np.array(meta[key], dtype=np.float64, copy=True)
    bad.reshape(-1)[-1] *= 1.0 + 1e-12      # exact check: a 1-ulp-scale change must be caught
    meta[key] = bad
    with pytest.raises(ValueError, match=f"metadata/{key}"):
        _check_cost_metadata(_write(tmp_path / "a.h5", meta), NET)


def test_der_count_mismatch_rejected(tmp_path):
    meta = dict(MATCH, der_cost_lin=np.full(3, 5.0))
    with pytest.raises(ValueError, match="der_cost_lin"):
        _check_cost_metadata(_write(tmp_path / "a.h5", meta), NET)


def test_partial_price_metadata_rejected(tmp_path):
    meta = {k: v for k, v in MATCH.items() if k != "der_cost_quad"}
    with pytest.raises(ValueError, match="incomplete price metadata"):
        _check_cost_metadata(_write(tmp_path / "a.h5", meta), NET)
