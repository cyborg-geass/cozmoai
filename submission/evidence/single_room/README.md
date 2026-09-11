# Single-Room Evidence

This folder contains lightweight generated evidence copied from the latest
validated run of:

```powershell
uv run python scripts/finalize_pipeline.py
```

Included evidence:

- `submission_manifest.json`
- `final_result.json`
- `measurement_evaluation.json`
- `room_envelope.json`
- `room_envelope.png`
- `room_geometry.json`
- `room_geometry.png`
- `walls.json`
- `openings_wall5.json`
- `wall5_opening_profile.csv`
- `opening_semantics/semantic_openings.json`
- `opening_semantics/opening_semantics_overview.png`
- `room_semantics/room_semantics.json`
- `room_semantics/contact_sheet.png`

Large point-cloud artifacts are regenerated under `outputs/single_room` and are
not committed to git.

The current opening semantic status is `no_candidates`; the current room
semantic status is `model_unavailable` in this local environment.
