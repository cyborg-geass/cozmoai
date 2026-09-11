# Cozmo AI Applied AI Submission

This repository contains a reproducible single-room reconstruction pipeline for the
provided Cozmo AI dataset capture:

`../cozmo-dataset/raw_dataset/single_room/c00a170fe1`

The implementation separates semantic perception from metrology. Optional
pretrained vision-language models infer room semantics from sampled RGB frames,
while calibrated RGB-D geometry and odometry reconstruct the scene and produce
physical measurements. Neural model scores are never used as substitutes for
metric uncertainty.

## Current Result

Latest deterministic finalization run:

| Metric | Result |
| --- | ---: |
| Dimension A | 2.2621 m |
| Dimension A 95% CI | [2.2337, 2.2904] m |
| Dimension B | 6.0136 m |
| Dimension B 95% CI | [5.9880, 6.0392] m |
| Reconstructed footprint area | 14.3138 m2 |
| Rectangular reference area | 13.6031 m2 |
| Floor residual P95 | 0.0254 m |
| Detected wall planes | 6 |
| Opening candidates on selected wall | 0 |
| Room semantics status | model_unavailable locally |

The rectangular reference area is the product of the two selected wall-to-wall
separations. The reported floor-plan area is the reconstructed quadrilateral
footprint.

## Setup

The project uses `uv` and Python 3.12.

```powershell
cd C:\Users\ankit\Documents\cozmoai_assignment\cozmo-ai
uv sync
```

The dataset is expected beside this repository:

```text
C:\Users\ankit\Documents\cozmoai_assignment\
  cozmo-ai\
  cozmo-dataset\
    raw_dataset\
      single_room\
        c00a170fe1\
```

## Reproduce The Pipeline

```powershell
uv run python scripts/finalize_pipeline.py
```

The finalizer validates the raw capture, clears known stale generated artifacts,
runs all eight pipeline stages, validates each stage output, and writes:

```text
outputs/single_room/final_result.json
outputs/single_room/submission_manifest.json
```

Generated outputs are intentionally ignored by git because they include point
cloud artifacts. Regenerate them with the command above.

## Optional AI Perception

Room semantics are exposed through:

```powershell
uv run python scripts/classify_room.py <capture_dir> <output_dir>
```

By default, the classifier tries to use `openai/clip-vit-base-patch32` from local
model cache only. If `torch`, `transformers`, or model weights are unavailable,
the command writes `status = "model_unavailable"` and still produces sampled
frame evidence. This keeps the geometry pipeline reproducible without network or
GPU access.

Useful flags:

```powershell
uv run python scripts/finalize_pipeline.py --disable-ai
uv run python scripts/finalize_pipeline.py --room-samples 16 --device cpu
uv run python scripts/finalize_pipeline.py --allow-model-download
```

## Repository Layout

```text
configs/                  Capture and assignment compliance matrices
docs/                     Technical report, final status, submission checklist
scripts/                  Canonical final pipeline scripts
scripts/archive_exploratory/
                          Investigation/debug scripts retained for traceability
src/cozmo_ai/             Reusable geometry and IO helpers
src/cozmo_ai/perception/  Optional semantic perception layer
submission/               GitHub-facing submission entry point and evidence
tests/                    Unit tests for geometry and pipeline validation
```

The final GitHub submission entry point is `submission/README.md`. The
`submission/evidence/single_room` folder contains lightweight generated evidence
that can be reviewed directly in GitHub. Large point clouds remain reproducible
under `outputs/single_room` and are intentionally not committed.

## Run Tests

```powershell
uv run python -m unittest discover -s tests
```

## Build A Submission Bundle

Run the finalizer first, then package the tracked code/docs and generated evidence:

```powershell
uv run python scripts/package_submission.py
```

The bundle is written to:

```text
outputs/cozmo_ai_submission_bundle.zip
```

## Important Scope Notes

The assignment asks for broader benchmark coverage than the available local
evidence supports. This repository is careful not to overclaim:

- Photo-only registration and drift gates are not evaluated.
- Ceiling height is provisional and not demonstrated against the 1.5 cm gate.
- Laser/tape ground truth is not available.
- Multi-room and repeat-room ground-truth benchmarks are not available.
- Damage detection is not evaluated because the required benchmark evidence is not
  present.

See `configs/compliance_matrix.json` and `docs/TECHNICAL_REPORT.md` for the
submission-facing details.
