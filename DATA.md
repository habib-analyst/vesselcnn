# DATA.md — dataset card for VesselCNN

## Source (the only source)

**VesselMNIST3D** — part of **MedMNIST v2** (Yang et al., *MedMNIST v2 —
A large-scale lightweight benchmark for 2D and 3D biomedical image
classification*, Scientific Data 2023). VesselMNIST3D derives from the
open-access **IntrA** dataset of 3D intracranial aneurysm models
(Yang et al., 2020): 103 complete brain-vessel models reconstructed
from MRA images, from which 1,694 healthy vessel segments and 215
aneurysm segments were generated automatically; non-watertight meshes
were repaired with PyMeshFix and voxelized with trimesh into
28×28×28 volumes.

- Official distribution used: Zenodo record 10519652 (the MedMNIST v2
  release file), fetched 2026-10-29 (Asia/Karachi) with
  `scripts/download_data.sh` (curl; the fetch is MD5-checked before
  use). The full file is redistributed in neither repository; the
  official **test split only** is bundled under `assets/` so the
  headline metrics can be re-derived from the release artifacts.
- **License: CC BY 4.0** (the `medmnist` package's dataset INFO and
  the MedMNIST v2 paper). Attribution: this card and the README
  citation. The MIT license of this repository covers the code and
  the trained weights, not the dataset.
- No other dataset was used. No private or clinical data.

## Exact file used (verified in the foreground at build time)

| File | Bytes | MD5 | SHA-256 |
|---|---|---|---|
| `vesselmnist3d.npz` | 398,349 | `b41fd4f7e7e2feedddb201585ecafa1b` | `cc00c6b0869bf1f68cf7c13e52e8585eb2f38991acab5ad7abe1af1814a83206` |

The MD5 equals the value published in the `medmnist` package's INFO
for this dataset; the SHA-256 was computed in the foreground from the
downloaded file.

## Splits, from the data itself

The source paper's 7:1:2 split, verified by loading the file (not
copied from documentation):

| Split | Volumes | Vessel (0) | Aneurysm (1) | Aneurysm share | Majority baseline |
|---|---|---|---|---|---|
| train | 1,335 | 1,185 | 150 | 0.1124 | 0.8876 |
| val | 191 | 169 | 22 | 0.1152 | 0.8848 |
| test | 382 | 339 | 43 | 0.1126 | 0.8874 |

- Volumes are uint8, shape (28, 28, 28), values in [0, 255]; labels
  are uint8 in {0, 1} with 0 = vessel, 1 = aneurysm.
- **Exact-duplicate check:** MD5 hashes of every volume were compared
  across splits in the foreground — zero volumes appear in more than
  one split. (This checks exact duplicates only; near-duplicates
  arising from adjacent segments of the same source vessel model
  cannot be excluded from the released arrays, because the release
  carries no per-volume source-model identifier. The 7:1:2 split was
  made by the dataset authors at the segment level, as documented in
  the MedMNIST v2 paper — read the headline metrics with that caveat.)
- **Sparsity:** 1.93% of train voxels are non-zero. Per-volume
  non-zero voxel counts in train range from 6 to 1,449 (median 389):
  some segments are tiny fragments. Occupancy is therefore part of
  the failure analysis in `assets/evaluation_results.json`.

## Preprocessing and normalization

- The only preprocessing is the contract in `vesselcnn.infer`:
  /255, then normalize with the train-split scalar mean/std computed
  with chunked float64 accumulation over all 1,335 train volumes:
  exact mean 0.01933, exact std 0.13768 (these are also the rounded
  contract constants; `scripts/train.py` asserts agreement to 1e-4).
- No resampling, cropping, or intensity clipping: the volumes are
  used exactly as released.
- Training augmentation (random octahedral transforms) is a
  training-time operation only; it is not part of this data card's
  preprocessing and is documented in `scripts/train.py` and the
  README.
- Class weighting in the loss (weights [0.5633, 4.4500] =
  N/(2·n_class)) is a training decision recorded in
  `assets/vesselcnn_metrics.json`, included here because it is part
  of how the imbalance above was handled: plain cross-entropy on an
  88.8%-majority split would pay the model to ignore the aneurysm
  class entirely.
