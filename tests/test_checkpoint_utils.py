import json

import pytest

from utils import checkpoint_utils
from utils.checkpoint_utils import COMPLETE, IN_PROGRESS, CheckpointManager, file_digest, fingerprint

STAGES = ["roads", "mapping", "secnet", "partition"]


@pytest.fixture
def manifest(tmp_path):
    return tmp_path / "interim" / "999_checkpoints.json"


@pytest.fixture
def output(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text("data\n")
    return path


def finish_all(manifest, params=None):
    ckpt = CheckpointManager(str(manifest), STAGES)
    for stage in STAGES:
        ckpt.start(stage, params or {"p": stage})
        ckpt.complete(stage, params or {"p": stage})


def test_fingerprint_is_stable_and_order_independent():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_fingerprint_changes_with_stage_version(monkeypatch):
    before = {stage: fingerprint({"a": 1}, stage) for stage in STAGES}
    monkeypatch.setitem(checkpoint_utils.STAGE_VERSIONS, "secnet", checkpoint_utils.STAGE_VERSIONS.get("secnet", 1) + 1)
    after = {stage: fingerprint({"a": 1}, stage) for stage in STAGES}
    assert after["secnet"] != before["secnet"]
    assert all(after[s] == before[s] for s in STAGES if s != "secnet")


def test_stage_version_bump_keeps_earlier_stages(manifest, output, monkeypatch):
    finish_all(manifest)
    monkeypatch.setitem(checkpoint_utils.STAGE_VERSIONS, "secnet", checkpoint_utils.STAGE_VERSIONS.get("secnet", 1) + 1)
    ckpt = CheckpointManager(str(manifest), STAGES)
    assert ckpt.status("roads", {"p": "roads"}, [output]) == COMPLETE
    assert ckpt.status("mapping", {"p": "mapping"}, [output]) == COMPLETE
    assert ckpt.status("secnet", {"p": "secnet"}, [output]) is None


def test_file_digest(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    b.write_text("x")
    assert file_digest(a) == file_digest(b)
    b.write_text("y")
    assert file_digest(a) != file_digest(b)


def test_no_manifest_means_recompute(manifest, output):
    ckpt = CheckpointManager(str(manifest), STAGES)
    assert ckpt.status("roads", {"p": "roads"}, [output]) is None


def test_completed_stages_are_reused(manifest, output):
    finish_all(manifest)
    ckpt = CheckpointManager(str(manifest), STAGES)
    for stage in STAGES:
        assert ckpt.status(stage, {"p": stage}, [output]) == COMPLETE


def test_changed_parameters_invalidate_stage(manifest):
    finish_all(manifest)
    ckpt = CheckpointManager(str(manifest), STAGES)
    assert ckpt.status("secnet", {"p": "other"}) is None


def test_missing_output_file_invalidates_stage(manifest, tmp_path):
    finish_all(manifest)
    ckpt = CheckpointManager(str(manifest), STAGES)
    assert ckpt.status("roads", {"p": "roads"}, [tmp_path / "missing.csv"]) is None


def test_recomputing_a_stage_invalidates_later_stages(manifest):
    finish_all(manifest)
    ckpt = CheckpointManager(str(manifest), STAGES)
    assert ckpt.status("roads", {"p": "roads"}) == COMPLETE
    ckpt.start("mapping", {"p": "mapping"})
    # later stages are stale in this run and dropped from the manifest for later runs
    assert ckpt.status("secnet", {"p": "secnet"}) is None
    saved = json.loads(manifest.read_text())["stages"]
    assert set(saved) == {"roads", "mapping"}
    assert saved["mapping"]["status"] == IN_PROGRESS


def test_interrupted_stage_reports_in_progress(manifest):
    ckpt = CheckpointManager(str(manifest), STAGES)
    ckpt.start("roads", {"p": "roads"})
    ckpt.complete("roads", {"p": "roads"})
    ckpt.start("mapping", {"p": "mapping"})  # run "crashes" here

    ckpt = CheckpointManager(str(manifest), STAGES)
    assert ckpt.status("roads", {"p": "roads"}) == COMPLETE
    assert ckpt.status("mapping", {"p": "mapping"}) == IN_PROGRESS
    assert ckpt.status("mapping", {"p": "changed"}) is None


def test_from_scratch_ignores_manifest(manifest, output):
    finish_all(manifest)
    ckpt = CheckpointManager(str(manifest), STAGES, from_scratch=True)
    assert all(ckpt.status(stage, {"p": stage}, [output]) is None for stage in STAGES)


def test_unreadable_manifest_is_ignored(manifest):
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{broken")
    assert CheckpointManager(str(manifest), STAGES).status("roads", {}) is None
