# Final Pipeline Status

## Reproducible command

```powershell
uv run python scripts/finalize_pipeline.py
```

The script uses `../cozmo-dataset/raw_dataset` and writes artifacts to
`outputs/single_room/`. It validates the raw capture, clears known generated
outputs before recomputing, validates every stage output, writes a unified final
result, and writes a submission manifest.

## Demonstrated single-room result

- Reconstructed footprint area: **14.3138 m2**
- Dimension A: **2.2621 m**, 95% CI **[2.2337, 2.2904] m**
- Dimension B: **6.0136 m**, 95% CI **[5.9880, 6.0392] m**
- Floor P95 residual: **0.0254 m**
- Wall-plane P95 residuals: approximately **2.5-2.8 cm**
- Detected wall planes: **6**
- Selected wall pairs: **1 <-> 5** and **3 <-> 6**
- Opening semantics: **no_candidates**, because the geometric opening detector
  found zero reliable candidates on the selected wall
- Room semantics: **model_unavailable** locally, with sampled-frame evidence and
  contact sheet generated

The **13.6031 m2** rectangular product is diagnostic only. The reported plan area
is the reconstructed quadrilateral footprint, **14.3138 m2**.

## Ceiling-height caveat

The ceiling capture supports an approximately **3.06 m** floor-to-ceiling separation,
but this is **not claimed as a passed 1.5 cm gate**. The available uncertainty is much
larger and no laser/tape ground truth is available.

## Dataset/compliance caveat

Requirements not supported by the supplied benchmark are explicitly marked
`not_evaluated` or `not_demonstrated` in `configs/compliance_matrix.json`.

The optional learned room classifier is integrated as a separate semantic layer.
It does not change geometry or measurement outputs. On this machine, optional AI
dependencies/model weights are unavailable, so the semantic section reports
`model_unavailable` rather than failing the finalizer.

The optional learned opening classifier is also integrated as a separate fusion
layer. It can label projected geometry candidates as door/window detections when
model dependencies are available, but dimensions stay geometry-only. The current
capture has no geometric opening candidates, so it reports `no_candidates` and
writes an overview image documenting that fallback.

## Verification

Latest local verification:

```powershell
uv run python -m unittest discover -s tests
uv run python scripts/finalize_pipeline.py
uv run python scripts/package_submission.py
```
