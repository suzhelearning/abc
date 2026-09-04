# Third-party and vendor assets

This branch keeps the ABC baseline license and adds the following components.

| Component | Location | Terms / provenance |
| --- | --- | --- |
| ABC baseline | repository root | Apache-2.0; see `LICENSE` |
| Wuji Hand retargeting runtime | `wuji_retargeting/` | MIT; see `abc_minimal/third_party/wuji-retargeting/LICENSE` and the vendored README |
| DINOv3 implementation | `abc_minimal/third_party/dinov3/`, adapted in `abc_minimal/dit.py` | Meta DINOv3 license; see `abc_minimal/third_party/dinov3/LICENSE.md` |
| MuJoCo / ABC simulation dependencies | Python environment | Respect each package's upstream license and terms |
| Tianji-Wuji2 URDF and STL meshes | `assets/tianji_wuji2/` (byte hashes in `SHA256SUMS`) | Vendor-provided research assets; redistribution permission is **not yet confirmed** |

The Tianji-Wuji2 files must not be pushed to a public remote until the vendor
confirms redistribution terms.  The repository `.gitignore` keeps the local
bundle out of ordinary staging; a future publication should replace this
notice with the vendor's exact license or move the assets to a separately
authorized artifact store.  The code remains simulation-only and does not
contain follower, motor, fieldbus, or physical safety-control interfaces.
