"""
Checkpointing of the pipeline stages in main.py.

A JSON manifest per region records, for every stage, a fingerprint of the inputs and
parameters it was run with and whether it finished. A stage's output files are reused
only when
    - the run is not started from scratch,
    - no earlier stage was recomputed in this run (outputs would be stale),
    - the manifest says the stage completed with the same fingerprint, and
    - all its output files exist.
Outputs written before the manifest existed are therefore regenerated once.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.logging_utils import LogManager

logger = LogManager.get_logger("checkpoint_utils")

CHECKPOINT_VERSION = 2

# Bump a stage's version when a code change invalidates its previously generated outputs
# (later stages are recomputed as well). Stages not listed are at version 1.
STAGE_VERSIONS = {
    "secnet": 2,  # more candidate transformers on short, crowded road links
}

COMPLETE = "complete"
IN_PROGRESS = "in_progress"


def file_digest(path: str) -> str:
    """SHA-256 of a file's contents, used to detect changed input files."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(params: Dict[str, Any], stage: str = None) -> str:
    """Stable hash of JSON-serialisable stage parameters and the stage's code version."""
    payload = json.dumps(
        {"version": STAGE_VERSIONS.get(stage, 1), "params": params}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CheckpointManager:
    """Track which pipeline stages can be reused from their saved outputs."""

    def __init__(self, manifest_file: str, stages: List[str], from_scratch: bool = False):
        """
        Args:
            manifest_file (str): Path to the JSON manifest
            stages (List[str]): Stage names in pipeline order
            from_scratch (bool): Ignore all saved outputs and recompute every stage
        """
        self.manifest_file = Path(manifest_file)
        self.stages = list(stages)
        self.from_scratch = from_scratch
        self._recomputing = from_scratch
        self._manifest = self._load()

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self.manifest_file.exists():
            return {}
        try:
            return json.loads(self.manifest_file.read_text()).get("stages", {})
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Ignoring unreadable checkpoint manifest {self.manifest_file}")
            return {}

    def _save(self):
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": CHECKPOINT_VERSION, "stages": self._manifest}, indent=2))
        os.replace(tmp, self.manifest_file)  # atomic: never leave a half-written manifest

    def status(
            self,
            stage: str,
            params: Dict[str, Any],
            files: Iterable[str] = (),
            adopt_existing: bool = False,
        ) -> Optional[str]:
        """
        Return COMPLETE if the stage's outputs can be reused, IN_PROGRESS if a previous run with
        the same parameters was interrupted in this stage, and None if it must start afresh.

        adopt_existing: accept output files that were written before checkpointing existed
            (no manifest entry for the stage) as complete. Use only for outputs that do not
            depend on code that has since changed, e.g. raw downloads.
        """
        if self._recomputing:
            return None
        files = list(files)
        entry = self._manifest.get(stage)
        if entry is None and adopt_existing and files and all(Path(f).exists() for f in files):
            logger.info(f"Adopting existing outputs of stage '{stage}' without a checkpoint")
            self.complete(stage, params)
            return COMPLETE
        if not entry or entry.get("fingerprint") != fingerprint(params, stage):
            return None
        if entry.get("status") == COMPLETE and not all(Path(f).exists() for f in files):
            logger.warning(f"Checkpoint for stage '{stage}' is missing output files; recomputing it")
            return None
        return entry.get("status")

    def start(self, stage: str, params: Dict[str, Any]):
        """Mark a stage as running; every later stage becomes stale and is recomputed."""
        self._recomputing = True
        for later in self.stages[self.stages.index(stage) + 1:]:
            self._manifest.pop(later, None)
        self._manifest[stage] = {"fingerprint": fingerprint(params, stage), "status": IN_PROGRESS}
        self._save()

    def complete(self, stage: str, params: Dict[str, Any]):
        """Mark a stage as finished; its outputs can be reused by later runs."""
        self._manifest[stage] = {"fingerprint": fingerprint(params, stage), "status": COMPLETE}
        self._save()
