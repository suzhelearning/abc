"""Build a Tianji/Wuji2 hammer scene from external, local robot/scan assets."""

import json

import tyro

from abc_sim.tianji_scene import SceneConfig, build_tianji_scene


if __name__ == "__main__":
    print(json.dumps(build_tianji_scene(tyro.cli(SceneConfig)), indent=2))
