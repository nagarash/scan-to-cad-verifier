# Scan-only CadQuery reconstruction

## Deliverables and execution

- `reconstruct.py`: standalone editable CadQuery construction, including embedded engraving outlines. No scan or external geometry is needed to rerun it.
- `reconstructed.step`: analytic B-rep solid, in millimeters.
- `reconstructed.stl`: triangulated solid; interpret STL units as millimeters.
- `reconstructed.png`: CAD preview.

Executed using Python 3.12, CadQuery 2.8.0 and trimesh. Run `python reconstruct.py` in an environment with `cadquery` and `trimesh`. Output files are written beside the script. Main dimensions are named parameters; individual feature dimensions remain editable in the construction code.

Only `outputs/reverse_engineering/ctc05/input/scan.ply` was read as geometry evidence. No reference CAD, benchmark metadata, existing renders, or other scans were used. The PLY contains 600,000 XYZ points, with no surface connectivity. Coordinates and original axes are retained. All dimensions below are millimeters.

## Measured features represented

| Feature | Scan-supported reconstruction |
|---|---|
| Flange | Radius 279.4, thickness 25.4; parallel flats at X = ±263.525 |
| Bolt pattern | Ten holes at approximately radius 222.25, with individually fitted centers embedded in source; absent locations along ±X are not invented |
| Bolt bores | Radius 9.525 from Z=0 to 12.7; radius 19.05 counterbores from Z=12.7 to 25.4 |
| Central housing | Radius 152.4; top Z=127; top 6.35 × 45° chamfer |
| Underside cavity | Radius 127, ending at Z=101.6; entrance radius 136.525 at Z=0 tapering to radius 127 at Z=9.525 |
| Mounting arms | Ends at X=±263.525, sides Y=±69.85, crest Z=54.61; crown is a radius-203.2 cylinder about the X axis centered at Y=0, Z=−146.05 |
| Open arm recesses | Rounded inward ends centered at X=±203.2, Y=0; radius 19.05; floor Z=45.72; open toward the outer ends |
| Four arm blind holes | X=±203.2 and ±234.95, Y=0; radius 3.96875; cylindrical bottoms Z=15.24, conical tips approximately Z=12.8553 |
| Upper central support | X=±63.5; forward boundary radius 127 centered at X=0,Y=−50.8; spans Z=127 to 279.4 |
| Inclined rear web | X=±88.9; tilt 12.5°; normal thickness 50.8; rear upper edge Y=−50.8 |
| Six side blind holes | Three from each X face, radius approximately 5.3581; cylinder depth 30.48, followed by approximately 3.2195-deep drill points. Y/Z centers: (−48.7109,171.45), (−40.9682,206.375), (−33.2255,241.3) |
| Shaft | Radius 31.75; top Z=482.6; 6.35 × 45° top chamfer; radius-6.35 root blend |
| Other blends | Radius-6.35 analytic blends at housing/arm feet, arm crowns/housing, and upper-support foot |
| Engraving | Recess on positive-X support face, floor X≈61.468 versus face X=63.5; approximately 2.032 deep. Two connected polygonal stroke outlines derived from scan points, not a substitute font |

The fitted bolt cylinders have radial residual standard deviations of approximately 0.005 mm. The crown and curved support fits also have typical residuals of approximately 0.003–0.005 mm. These describe point scatter around local fits, not manufacturing tolerances or certified scanner accuracy.

## Assumptions and unresolved geometry

1. Analytic planes, cylinders, cones, circular blends and nominal rounded dimensions replace noisy measured surfaces. Symmetry and shared radii are imposed where supported by matching point distributions. This is a design-intent reconstruction, not a triangulation of the scan.
2. The engraving depth is well supported, but its precise outline is unresolved. Alpha-shape outlines were simplified to polygons with approximately 0.3 mm simplification tolerance and small sampling artifacts removed. A local sampled deviation reaches 1.224 mm near a rounded stroke corner. Exact font, original lettering sketch and tiny edge radii are not claimed.
3. Small patches where the mounting-arm side blends meet the round housing and base are approximate. The CAD fillet construction does not reproduce every original blend junction: sampled local residuals reach approximately 0.823 mm. These patches are retained as smooth analytic transitions rather than omitted, but the original blend network remains unresolved.
4. Drill-point cones are inferred from observed shrinking radial sections. Threads, material, fits, tolerances and unobserved sealed internal features cannot be established from these XYZ surface samples. No threads or hidden cavities were invented.
5. The model assumes one continuous solid. The scan provides no evidence of separate assembly interfaces.
6. STL tessellation uses 0.08 mm linear and 0.08 rad angular settings. Duplicate/zero-area seam facets are removed after export, with vertex merging at seven decimal places. This cleanup affects tessellation only; STEP geometry is unchanged.

No detected bore, counterbore, blind drill point, open recess or underside cavity was intentionally omitted.

## Verification

- Executed the Python source and exported both requested formats.
- CadQuery/OpenCascade validity check: valid, one connected solid.
- STEP was reimported for exact distance checks.
- Final STL was checked for watertightness after seam-facet cleanup.
- Exact unsigned distances from a reproducible random sample of 6,000 of the 600,000 scan points to the STEP B-rep (seed 42):

| Statistic | Distance (mm) |
|---|---:|
| Median | 0.000258 |
| 90th percentile | 0.006419 |
| 95th percentile | 0.008277 |
| 99th percentile | 0.013134 |
| Maximum sampled | 1.224005 |

These are one-way, sampled surface distances, not a whole-model Hausdorff bound or a guarantee about unobserved surfaces. Exact distance results are saved in `exact_distances.npz`; `exact_check.py` reproduces the check. The initial `deviations.png` was only a coarse, sampled-surface diagnostic and is not the final accuracy measurement.

## Inspection evidence

`projections.png`, `slices.png`, `details.png`, and `upper.png` show point-cloud measurements. `measurements.txt` contains section bounds and fitted bolt circles. Inspection scripts are retained for reproducibility; their input is exclusively the supplied PLY (or the points array parsed from it).
