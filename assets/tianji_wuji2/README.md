# Tianji-Wuji2 model assets

This directory is the authoritative URDF-first source for the SPD-VR digital
twin. The generated MJCF files and joint manifests must be derived from
`tianji_wuji2.urdf`; do not hand-maintain a second robot structure.

These files were supplied for the research integration and do not include an
explicit redistribution license. Confirm the Tianji/Wuji2 vendor terms before
publishing or redistributing this directory.

The supplied URDF declares the robot name
`Marvin M6-S-CCS-696-V4.0_Base_and_Stand_Asm urdf`. TianJi's public company
page describes a Marvin-series product line, which is provenance context only;
it does not grant redistribution rights for this local URDF/STL bundle.
See <https://gentorobotics.ai/about-us/> and obtain written vendor terms
before any public asset release.

`SHA256SUMS` records the byte hashes of the supplied URDF, meshes, and this
README. Verify it from the repository root with:

```bash
cd assets/tianji_wuji2 && sha256sum -c SHA256SUMS
```
