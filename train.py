"""Launch ABC training.

Trains ABC-DiT by default, ABC-VLA with ``--policy vla``, or Tianji/Wuji2 SPD
with ``--policy spd``. VLA takes ``--vla-model.backbone.checkpoint``; SPD takes
``--spd-data.root`` and ``--spd-data.dino-checkpoint``. See the package README
for policy recipes and ``python train.py --help`` for all flags.
"""

import tyro

from abc_minimal.config import TrainConfig
from abc_minimal.train_loop import main as train


def main():
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
