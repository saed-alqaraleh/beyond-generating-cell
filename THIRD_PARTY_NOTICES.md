# Third-party datasets and implementations

The experiments described in this repository use external research resources. This repository does **not** bundle the original source datasets, pretrained baseline checkpoints, or the baseline implementations listed below.

| Resource | Upstream source | Relevant rights information |
| --- | --- | --- |
| KITTI-360 | [KITTI-360 project](https://www.cvlibs.net/datasets/kitti-360/) | The dataset providers state that their datasets and benchmarks are distributed under Creative Commons Attribution–NonCommercial–ShareAlike 3.0 (CC BY-NC-SA 3.0). Registered access and the dataset website's additional terms apply. |
| KITTI360Pose and Text2Pos | [Text2Pos-CVPR2022](https://github.com/mako443/Text2Pos-CVPR2022) | Cite the Text2Pos publication and obtain the benchmark inputs from the official authors. No general permissive software license was established for this upstream repository in the release audit; public access does not imply permission to relicense its code. |
| Text2Loc | [Official Text2Loc repository](https://github.com/Yan-Xia/Text2Loc) | Obtain original code and model weights from the authors. No general permissive software license was established for this upstream repository in the release audit. |
| Text2Loc++ | [Official Text2Loc++ repository](https://github.com/anslt/Text2LocPP) | The upstream source repository declares the MIT License, copyright © 2026 Letian Shi. This does not transfer copyright in baseline code or datasets to this project. |
| KITTI-360 utility scripts | [kitti360Scripts](https://github.com/autonomousvision/kitti360Scripts) | The upstream scripts use MIT; KITTI-360 source datasets remain under their separate dataset license. |

## Derived experimental artifacts

The bit-packed query-to-cell compatibility records, frozen rankings, candidate identifiers, constrained-chain outputs and statistical tables included here were produced by the experiments accompanying the article. Where any released artifact constitutes an adaptation or redistribution of content protected by the KITTI-360 or KITTI360Pose terms, those upstream conditions continue to apply, including attribution, noncommercial use and share-alike obligations as applicable. The presence of a research artifact here is not a grant of rights to original KITTI-360 or baseline content.

The source article and the external dataset and baseline papers must be cited when their materials are used. For author-developed evaluation code, a code license is distinct from the license governing underlying dataset-derived material. See the root `README.md` and `DATA_PROVENANCE.md` for the exact reproducibility scope.

No claim is made that the baseline authors endorse this repository.
