# Final Pipeline Status

## Reproducible command

```powershell
uv run python scripts/finalize_pipeline.py
```

The script uses `../cozmo-dataset/raw_dataset` and writes artifacts to `outputs/`.

## Demonstrated single-room result

- Reconstructed footprint area: **14.4260 m²**
- Dimension A: **2.2496 m**, 95% CI **[2.2201, 2.2790] m**
- Dimension B: **6.0050 m**, 95% CI **[5.9787, 6.0312] m**
- Wall-plane P95 residuals: approximately **2.3–2.9 cm**

The **13.5085 m²** rectangular product is diagnostic only. The reported plan area is
the reconstructed quadrilateral footprint, **14.4260 m²**.

## Ceiling-height caveat

The ceiling capture supports an approximately **3.06 m** floor-to-ceiling separation,
but this is **not claimed as a passed 1.5 cm gate**. The available uncertainty is much
larger and no laser/tape ground truth is available.

## Dataset/compliance caveat

Requirements not supported by the supplied benchmark are explicitly marked
`not_evaluated` or `not_demonstrated` in `configs/compliance_matrix.json`.
