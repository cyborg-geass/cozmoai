# Technical Report

## Summary

This submission implements the depth/LiDAR tier on the supplied single-room
capture `single_room/c00a170fe1` and adds an optional learned perception layer
for RGB semantics. The reproducible pipeline builds a global point cloud from
depth frames and odometry, fits floor and wall planes, selects a
trajectory-constrained room envelope, reports model-based uncertainty for the two
primary wall-to-wall room dimensions, and writes a final result that separates
semantic confidence from metric uncertainty.

The current run reconstructs a quadrilateral footprint area of **14.3138 m2** with
primary separations of **2.2621 m** and **6.0136 m**. The result is not claimed as
ground-truth accuracy because the local dataset does not include tape/laser
measurements.

## Dataset And Capture

- Capture: `../cozmo-dataset/raw_dataset/single_room/c00a170fe1`
- RGB video: `rgb.mp4`
- Depth frames: 1715 PNG frames
- Confidence frames: 1715 PNG frames
- Pose stream: `odometry.csv`
- IMU stream: `imu.csv`
- Camera calibration: `camera_matrix.csv`

The finalizer validates that the required files exist and that depth/confidence
frame counts match before it computes artifacts.

## Pipeline

1. `scripts/build_pointcloud.py` back-projects depth frames using calibration and
   transforms them into the odometry frame.
2. `scripts/detect_floor.py` selects the floor from horizontal RANSAC candidates
   using vertical position, support, and footprint evidence.
3. `scripts/detect_walls.py` fits vertical planes above the detected floor, removes
   duplicate planes, and saves per-wall support clouds.
4. `scripts/build_room_geometry.py` builds a floor-frame representation of wall
   families, parallel pairs, and intersections.
5. `scripts/select_room_envelope.py` chooses the most plausible enclosing wall
   pairs using wall support, overlap, and trajectory containment.
6. `scripts/evaluate_measurements.py` computes plane residuals and propagates
   model-based uncertainty for the selected dimensions.
7. `scripts/detect_openings.py` profiles a selected wall and reports opening
   candidates only when supported by observed geometry.
8. `scripts/classify_room.py` samples RGB frames and runs optional zero-shot room
   classification through a lazily loaded vision-language model adapter. If the
   model or dependencies are unavailable, it records `model_unavailable` without
   failing the geometry pipeline.

`scripts/finalize_pipeline.py` is the canonical runner. It removes known stale
outputs before each run and validates every required artifact as soon as it is
produced.

## Quantitative Results

| Quantity | Value |
| --- | ---: |
| Downsampled global point cloud | 328,668 points |
| Floor support | 59,789 points |
| Floor residual P95 | 0.0254 m |
| Wall count | 6 |
| Dimension A | 2.2621 m |
| Dimension A 95% CI | [2.2337, 2.2904] m |
| Dimension B | 6.0136 m |
| Dimension B 95% CI | [5.9880, 6.0392] m |
| Rectangular reference area | 13.6031 m2 |
| Rectangular reference area 95% CI | [13.4229, 13.7833] m2 |
| Reconstructed footprint area | 14.3138 m2 |
| Trajectory inside selected envelope | 72.3% |
| Opening candidates on wall 5 | 0 |
| Room semantics status | model_unavailable locally |
| Sampled RGB frames for semantics | 15 decoded from 16 requested |

The rectangular reference area is included only as a diagnostic uncertainty
calculation. The submitted plan area is the reconstructed quadrilateral footprint
from the selected wall intersections.

## Learned Perception

The perception layer lives under `src/cozmo_ai/perception`. It contains typed
semantic schemas, deterministic RGB frame sampling, prompt ensembling, room-level
probability aggregation, and a lazy zero-shot model adapter. The adapter uses a
CLIP-compatible Hugging Face model when optional dependencies and weights are
available, but default execution does not require network access.

On this machine, optional AI dependencies are not installed, so the current
artifact reports `status = "model_unavailable"`, `label = "unknown"`, and
`confidence = 0.0`. The sampled frame indices and a contact-sheet evidence image
are still generated so the evaluator can verify what the semantic layer would
inspect.

Semantic confidence is similarity-derived model confidence. It is not a 95%
confidence interval and is not used for wall lengths, area, opening dimensions,
or any other metric output.

## Confidence And Calibration

Dimension intervals are derived from robust wall-plane residuals and first-order
uncertainty propagation. They are useful as model-based confidence intervals, but
they are not calibrated absolute accuracy claims. No ground-truth tape/laser
measurements are available in the local dataset.

The reconstructed quadrilateral footprint currently reports no area confidence
interval because corner covariance is not computed. The rectangular reference area
has a propagated interval and is kept separate to avoid overclaiming.

## Openings

The opening detector runs deterministically on wall 5 and writes both a JSON
artifact and wall profile CSV. The current pass reports zero reliable opening
candidates. This is intentional: the available geometry does not support a
height/width claim robustly enough for final reporting.

## Ceiling Height

Exploratory ceiling scripts estimate an approximate floor-to-ceiling separation of
3.06 m, but this is marked `not_demonstrated`. The evidence does not support the
assignment's 1.5 cm ceiling-height gate, and there is no ground truth for
calibration.

## Compliance Notes

The provided local evidence does not support all assignment tiers. The compliance
matrix is deliberately conservative:

- Photo-only registration and drift are not evaluated.
- Multi-room and repeat-room ground-truth benchmarks are not evaluated.
- Damage detection is not evaluated.
- Laser/tape ground truth is not available.
- Ceiling height is provisional, not demonstrated.

This conservative reporting is preferable to overstating performance on gates that
cannot be measured from the supplied local dataset.

## Reproducibility

Run:

```powershell
uv run python -m unittest discover -s tests
uv run python scripts/finalize_pipeline.py
uv run python scripts/package_submission.py
```

Primary generated artifacts are listed in
`outputs/single_room/submission_manifest.json`. The unified final result is
`outputs/single_room/final_result.json`. The packaging script writes
`outputs/cozmo_ai_submission_bundle.zip`.
