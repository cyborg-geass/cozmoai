# Final Pipeline Status

## Reproducible command

```powershell
uv run python scripts/finalize_pipeline.py
```

The script uses `../cozmo-dataset/raw_dataset` and writes artifacts to
`outputs/single_room/`. It validates the raw capture, clears known generated
outputs before recomputing, validates every stage output, and writes a submission
manifest.

## Demonstrated single-room result

- Reconstructed footprint area: **14.3138 m2**
- Dimension A: **2.2621 m**, 95% CI **[2.2337, 2.2904] m**
- Dimension B: **6.0136 m**, 95% CI **[5.9880, 6.0392] m**
- Floor P95 residual: **0.0254 m**
- Wall-plane P95 residuals: approximately **2.5-2.8 cm**
- Detected wall planes: **6**
- Selected wall pairs: **1 <-> 5** and **3 <-> 6**

The **13.6031 m2** rectangular product is diagnostic only. The reported plan area
is the reconstructed quadrilateral footprint, **14.3138 m2**.

## Ceiling-height caveat

The ceiling capture supports an approximately **3.06 m** floor-to-ceiling separation,
but this is **not claimed as a passed 1.5 cm gate**. The available uncertainty is much
larger and no laser/tape ground truth is available.

## Dataset/compliance caveat

Requirements not supported by the supplied benchmark are explicitly marked
`not_evaluated` or `not_demonstrated` in `configs/compliance_matrix.json`.

## Verification

Latest local verification:

```powershell
uv run python -m unittest discover -s tests
uv run python scripts/finalize_pipeline.py
uv run python scripts/package_submission.py
```
