"""resolve_checkpoint_path: latest-epoch selection and explicit-path passthrough."""

from core.utils.checkpoint import resolve_checkpoint_path


def _make(root, *names):
    for name in names:
        (root / name).mkdir()


def test_latest_epoch_is_numeric_not_lexical(tmp_path):
    _make(tmp_path, "checkpoint_epoch_5", "checkpoint_epoch_10", "checkpoint_epoch_100")
    assert resolve_checkpoint_path(tmp_path).name == "checkpoint_epoch_100"


def test_ten_beats_five(tmp_path):
    _make(tmp_path, "checkpoint_epoch_5", "checkpoint_epoch_10")
    assert resolve_checkpoint_path(tmp_path).name == "checkpoint_epoch_10"


def test_non_numeric_suffix_is_ignored(tmp_path):
    _make(tmp_path, "checkpoint_epoch_5", "checkpoint_epoch_old")
    assert resolve_checkpoint_path(tmp_path).name == "checkpoint_epoch_5"


def test_directory_with_config_is_returned_as_is(tmp_path):
    ckpt = tmp_path / "checkpoint_epoch_5"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    _make(ckpt, "checkpoint_epoch_10")
    assert resolve_checkpoint_path(ckpt) == ckpt


def test_no_candidates_returns_input(tmp_path):
    assert resolve_checkpoint_path(tmp_path) == tmp_path
