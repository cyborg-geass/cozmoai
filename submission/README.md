# GitHub Submission Entry Point

Use this folder as the reviewer-facing entry point when the repository URL is the
submission.

## Start Here

- Final results and setup: `../README.md`
- Technical report: `../docs/TECHNICAL_REPORT.md`
- Compliance matrix: `../configs/compliance_matrix.json`
- Capture/device matrix: `../configs/capture_matrix.json`
- Lightweight generated evidence: `evidence/single_room`

## Reproduce

```powershell
uv run python -m unittest discover -s tests
uv run python scripts/finalize_pipeline.py
```

The finalizer refreshes `submission/evidence/single_room` with JSON, PNG, and CSV
evidence that is suitable for GitHub review. Large point-cloud files are generated
under `outputs/single_room` and are intentionally not tracked.
