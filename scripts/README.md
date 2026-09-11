# Scripts

The root of this folder contains the final reproducible pipeline.

## Canonical Workflow

```powershell
uv run python scripts/finalize_pipeline.py
uv run python scripts/classify_room.py <capture_dir> <output_dir>
uv run python scripts/classify_openings.py <capture_dir> <openings_json> <output_dir>
uv run python scripts/package_submission.py
```

## Final Pipeline Scripts

- `build_pointcloud.py`
- `detect_floor.py`
- `detect_walls.py`
- `build_room_geometry.py`
- `select_room_envelope.py`
- `evaluate_measurements.py`
- `detect_openings.py`
- `classify_openings.py`
- `classify_room.py`
- `finalize_pipeline.py`
- `package_submission.py`

Exploratory and debugging scripts are retained in `archive_exploratory/` for
traceability.
