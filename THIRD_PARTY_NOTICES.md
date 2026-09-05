# Third-party and vendor assets

This branch keeps the ABC baseline license and adds the following components.

| Component | Location | Terms / provenance |
| --- | --- | --- |
| ABC baseline | repository root | Apache-2.0; see `LICENSE` |
| Wuji Hand retargeting runtime | `wuji_retargeting/` | MIT; see `abc_minimal/third_party/wuji-retargeting/LICENSE` and the vendored README |
| Wuji public model/retargeting upstreams | [wuji-description](https://github.com/wuji-technology/wuji-description), [wuji-hand-description](https://github.com/wuji-technology/wuji-hand-description), [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) | The official repositories display an MIT license. This is provenance evidence for the corresponding upstream projects only; it is not permission for the local combined Tianji-Wuji2 bundle. |
| DINOv3 implementation | `abc_minimal/third_party/dinov3/`, adapted in `abc_minimal/dit.py` | Meta DINOv3 license; see [`LICENSE.md`](abc_minimal/third_party/dinov3/LICENSE.md). The intended official ViT-B/16 source is [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m); weights are not bundled, and the access page requires the user to accept its gated terms. |
| MuJoCo / ABC simulation dependencies | Python environment | Respect each package's upstream license and terms |
| Tianji-Wuji2 URDF and STL meshes | `assets/tianji_wuji2/` (byte hashes in `SHA256SUMS`) | Vendor-provided research assets; redistribution permission is **not yet confirmed** |

The Tianji-Wuji2 files must not be pushed to a public remote until the vendor
confirms redistribution terms.  The repository `.gitignore` keeps the local
bundle out of ordinary staging; a future publication should replace this
notice with the vendor's exact license or move the assets to a separately
authorized artifact store.  The public Wuji repositories above do not, by
themselves, establish terms for the supplied Tianji arm meshes or the local
combined URDF.  The code remains simulation-only and does not contain
follower, motor, fieldbus, or physical safety-control interfaces.
