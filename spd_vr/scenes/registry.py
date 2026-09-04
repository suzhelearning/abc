"""SPD's six-scene, 17-task deterministic registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .scene_builder import ProceduralSceneBuilder, SceneBuildResult


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt: str
    target_duration_s: float
    build: Callable[[int], SceneBuildResult]
    reset: Callable[[int], SceneBuildResult]
    score_debug: Callable[[SceneBuildResult], dict[str, float | int | str]]
    scene: str
    table2_episodes: int
    table2_minutes: int

    @property
    def qualified_name(self) -> str:
        return f"{self.scene}/{self.name}"


# Source of truth: SPD paper Table 2.  The per-episode target is exactly
# 60 seconds/minute × total minutes / episode count.
TABLE2_STATS = {
    ("jenga", "hollow_tower"): (92, 567),
    ("jenga", "tower"): (87, 473),
    ("jenga", "dominos"): (107, 471),
    ("jenga", "criss_cross"): (103, 386),
    ("jenga", "handover_lr"): (96, 172),
    ("jenga", "handover_rl"): (72, 109),
    ("spelling_blocks", "spelling"): (168, 587),
    ("spelling_blocks", "sort_and_unload"): (25, 144),
    ("spelling_blocks", "pyramid"): (32, 136),
    ("spelling_blocks", "vowel_consonant_sort"): (50, 109),
    ("mugs", "hang_mug"): (406, 491),
    ("dishes", "rack_dishes"): (129, 285),
    ("dishes", "plate_dishes"): (79, 109),
    ("cups", "pyramid"): (44, 109),
    ("cups", "stack_two_threes"): (46, 67),
    ("cups", "unstack"): (30, 47),
    ("bottles", "toss_in_bin"): (350, 253),
}
PROMPTS = {
    ("jenga", "hollow_tower"): "Build a hollow Jenga tower.",
    ("jenga", "tower"): "Build a stable Jenga tower.",
    ("jenga", "dominos"): "Arrange the blocks as a domino chain.",
    ("jenga", "criss_cross"): "Build a criss-cross Jenga tower.",
    ("jenga", "handover_lr"): "Hand the block from the left hand to the right.",
    ("jenga", "handover_rl"): "Hand the block from the right hand to the left.",
    ("spelling_blocks", "spelling"): "Spell the requested word with letter blocks.",
    ("spelling_blocks", "sort_and_unload"): "Sort the letter blocks and unload them.",
    ("spelling_blocks", "pyramid"): "Build a pyramid from the letter blocks.",
    ("spelling_blocks", "vowel_consonant_sort"): "Sort letters into vowels and consonants.",
    ("mugs", "hang_mug"): "Hang the mug on the mug tree.",
    ("dishes", "rack_dishes"): "Place the plates into the dish rack.",
    ("dishes", "plate_dishes"): "Arrange the plates into a dish stack.",
    ("cups", "pyramid"): "Build a pyramid from the cups.",
    ("cups", "stack_two_threes"): "Build two stacks of three cups.",
    ("cups", "unstack"): "Unstack the nested cups.",
    ("bottles", "toss_in_bin"): "Toss every bottle into the bin.",
}


def _build(scene: str, task: str, seed: int) -> SceneBuildResult:
    return ProceduralSceneBuilder(scene, task, seed).build()


def _score(result: SceneBuildResult) -> dict[str, float | int | str]:
    return {
        "scene": result.scene,
        "task": result.task,
        "objects": len(result.objects),
        "candidate": result.candidate,
        "contact_group_count": len({item.contact_group for item in result.objects}),
    }


def _task(scene: str, task: str) -> TaskSpec:
    episodes, minutes = TABLE2_STATS[(scene, task)]
    duration = 60.0 * float(minutes) / float(episodes)
    builder = lambda seed: _build(scene, task, seed)
    return TaskSpec(
        name=task,
        prompt=PROMPTS[(scene, task)],
        target_duration_s=duration,
        build=builder,
        reset=builder,
        score_debug=_score,
        scene=scene,
        table2_episodes=episodes,
        table2_minutes=minutes,
    )


TASKS: tuple[TaskSpec, ...] = tuple(_task(scene, task) for scene, task in TABLE2_STATS)
TASK_REGISTRY = {spec.qualified_name: spec for spec in TASKS}
SCENES = tuple(dict.fromkeys(spec.scene for spec in TASKS))


def get_task(scene: str, task: str | None = None) -> TaskSpec:
    if task is None and "/" in scene:
        scene, task = scene.split("/", 1)
    if task is None:
        matches = [spec for spec in TASKS if spec.scene == scene]
        if len(matches) != 1:
            raise KeyError(f"task is required for scene {scene!r}")
        return matches[0]
    try:
        return TASK_REGISTRY[f"{scene}/{task}"]
    except KeyError as exc:
        raise KeyError(f"unknown SPD task: {scene}/{task}") from exc


def iter_tasks() -> Iterable[TaskSpec]:
    return iter(TASKS)


__all__ = [
    "PROMPTS",
    "SCENES",
    "TABLE2_STATS",
    "TASKS",
    "TASK_REGISTRY",
    "TaskSpec",
    "get_task",
    "iter_tasks",
]
