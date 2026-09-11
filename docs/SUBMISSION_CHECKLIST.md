# Submission Checklist

## Included

- Reproducible pipeline runner: `scripts/finalize_pipeline.py`
- Room semantic CLI: `scripts/classify_room.py`
- Opening semantic CLI: `scripts/classify_openings.py`
- Submission bundle builder: `scripts/package_submission.py`
- Source package: `src/cozmo_ai`
- Optional perception package: `src/cozmo_ai/perception`
- Pipeline scripts: `scripts`
- Unit tests: `tests`
- Technical report: `docs/TECHNICAL_REPORT.md`
- Pipeline status summary: `docs/FINAL_PIPELINE_STATUS.md`
- Compliance matrix: `configs/compliance_matrix.json`
- GitHub submission entry point: `submission/README.md`
- Lightweight generated evidence: `submission/evidence/single_room`
- Full generated manifest: `outputs/single_room/submission_manifest.json`
- Unified final result: `outputs/single_room/final_result.json`

## Not In The Main Path

- Exploratory/debug scripts are retained in `scripts/archive_exploratory`.
- Large point-cloud artifacts are regenerated in `outputs/single_room` and are not
  tracked in git.

## Verification Commands

```powershell
uv run python -m unittest discover -s tests
uv run python scripts/finalize_pipeline.py
uv run python scripts/package_submission.py
```

## Current Submission Constraints

- Photo-only tier is not evaluated.
- Ceiling-height 1.5 cm gate is not demonstrated.
- Multi-room and repeat-room benchmark gates are not evaluated.
- Damage detection is not evaluated.
- Ground-truth absolute accuracy is not available.
- Opening semantics are implemented, but the current capture has zero geometric
  opening candidates, so the semantic output is `no_candidates`.
- Room classification model weights/dependencies are not installed locally, so
  current semantic output is `model_unavailable`.

These are data/evidence constraints, not hidden implementation passes.
