# Single-Room Evidence

This folder contains lightweight generated evidence copied from the latest
validated run of:

```powershell
uv run python scripts/finalize_pipeline.py
```

Included evidence:

- `submission_manifest.json`
- `measurement_evaluation.json`
- `room_envelope.json`
- `room_envelope.png`
- `room_geometry.json`
- `room_geometry.png`
- `walls.json`
- `openings_wall5.json`
- `wall5_opening_profile.csv`

Large point-cloud artifacts are regenerated under `outputs/single_room` and are
not committed to git.
