"""Atomic dual-hand Wuji retargeting with per-side HOLD semantics.

``WujiRetargetAdapter`` is the production adapter used by the viewer.  This
module exposes a small pair abstraction for offline tools and callers that
need to inspect each hand's validity/reason without coupling to the viewer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from .pico_hands import HandFrameError, PicoHandFrame, PicoHandsInput, PICO_TO_MEDIAPIPE
from .manifest import load_manifest
from .robot import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS

try:
    # Keep the constructor at module scope so replay/tests can replace it
    # without importing vendor code in the real-time loop.
    from wuji_retargeting import Retargeter
except ImportError:  # pragma: no cover - dependency/package failure path
    Retargeter = None  # type: ignore[assignment,misc]


class HandHoldReason(str, Enum):
    NONE = "none"
    INACTIVE = "inactive"
    SOLVER_FAILURE = "solver_failure"
    STALE = "stale"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RetargetedHands:
    tracking_epoch: int
    sequence_id: int
    left_qpos: np.ndarray
    right_qpos: np.ndarray
    left_valid: bool
    right_valid: bool
    left_hold_reason: HandHoldReason
    right_hold_reason: HandHoldReason

    def __post_init__(self) -> None:
        for name in ("left_qpos", "right_qpos"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (20,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 20-D vector")
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "left_hold_reason", HandHoldReason(self.left_hold_reason))
        object.__setattr__(self, "right_hold_reason", HandHoldReason(self.right_hold_reason))


def qpos_reorder_perm(
    src_joint_names: Sequence[str], dst_joint_names: Sequence[str]
) -> np.ndarray:
    """Return a strict source-to-destination index permutation."""

    source, destination = list(src_joint_names), list(dst_joint_names)
    if len(source) != len(destination) or not source:
        raise ValueError("source and destination joint names must have equal non-zero length")
    if len(set(source)) != len(source) or len(set(destination)) != len(destination):
        raise ValueError("joint names must be unique")
    if set(source) != set(destination):
        raise ValueError(
            f"joint-name mapping failed: missing={sorted(set(destination) - set(source))}, "
            f"extra={sorted(set(source) - set(destination))}"
        )
    source_index = {name: index for index, name in enumerate(source)}
    return np.asarray([source_index[name] for name in destination], dtype=np.int64)


def _source_names(retargeter: Any, side: str) -> list[str]:
    robot = getattr(getattr(retargeter, "optimizer", None), "robot", None)
    names = getattr(robot, "dof_joint_names", None)
    if names is None:
        raise ValueError(f"{side} retargeter does not expose dof_joint_names")
    names = list(names)
    if len(names) != 20:
        raise ValueError(f"{side} retargeter must expose exactly 20 joints")
    return names


def _source_limits(retargeter: Any) -> np.ndarray:
    robot = getattr(getattr(retargeter, "optimizer", None), "robot", None)
    limits = getattr(robot, "joint_limits", None)
    if limits is None:
        return np.full((20, 2), (-np.inf, np.inf), dtype=np.float64)
    value = np.asarray(limits, dtype=np.float64)
    if value.shape != (20, 2) or np.any(~np.isfinite(value)) or np.any(value[:, 0] > value[:, 1]):
        raise ValueError("retargeter joint limits must be finite [20,2] bounds")
    return value


def _resolve_path(config: Mapping[str, Any], key: str) -> Path:
    raw = (config.get("optimizer") or {}).get(key)
    if not raw:
        raise ValueError(f"optimizer.{key} is required for strict Wuji mapping")
    path = Path(str(raw))
    if not path.is_absolute():
        yaml_dir = config.get("__yaml_dir")
        if not yaml_dir:
            raise ValueError(f"optimizer.{key} is relative but __yaml_dir is missing")
        path = Path(str(yaml_dir)) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"optimizer.{key} not found: {path}")
    return path


def mjcf_actuator_joint_names(path: str | Path) -> list[str]:
    """Resolve the actuator target joint names in official MJCF order."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency setup
        raise ImportError("mujoco is required to resolve Wuji actuator order") from exc
    model = mujoco.MjModel.from_xml_path(str(path))
    names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise ValueError(f"MJCF actuator {actuator_id} is not joint-backed")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"MJCF actuator {actuator_id} has no named joint")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("MJCF actuator joint names are duplicated")
    return names


def _config_from(value: Any) -> Any:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return value


def _manifest_hand_contract(
    manifest_path: str | Path, urdf_path: str | Path
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate compiler provenance and return manifest hand joint orders."""

    document = load_manifest(Path(manifest_path).resolve())
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("manifest source metadata is missing")
    urdf = Path(urdf_path).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    declared = source.get("urdf")
    if declared and Path(str(declared)).name != urdf.name:
        raise ValueError(f"manifest URDF path mismatch: declared={declared!r}, actual={urdf}")
    expected_urdf_hash = source.get("urdf_sha256")
    if not isinstance(expected_urdf_hash, str) or hashlib.sha256(urdf.read_bytes()).hexdigest() != expected_urdf_hash:
        raise ValueError("authoritative URDF hash mismatch")
    expected_manifest_hash = document.get("manifest_sha256")
    normalized = dict(document)
    normalized["manifest_sha256"] = ""
    actual_manifest_hash = hashlib.sha256(
        yaml.safe_dump(normalized, sort_keys=True, allow_unicode=True).encode("utf-8")
    ).hexdigest()
    if not isinstance(expected_manifest_hash, str) or actual_manifest_hash != expected_manifest_hash:
        raise ValueError("model manifest hash mismatch")
    hand_order = document.get("hand_joint_order")
    entries = document.get("joints")
    actuator_order = document.get("actuator_order")
    if not isinstance(hand_order, Mapping) or not isinstance(entries, list) or not isinstance(actuator_order, list):
        raise ValueError("manifest hand/actuator joint order is missing")
    by_actuator = {
        entry.get("actuator"): entry
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("actuator"), str)
    }

    def side_order(side: str) -> list[str]:
        names = hand_order.get(side)
        if not isinstance(names, list) or len(names) != 20 or len(set(names)) != 20:
            raise ValueError(f"manifest {side} hand_joint_order must contain 20 unique joints")
        ordered: list[str] = []
        for actuator in actuator_order:
            entry = by_actuator.get(actuator)
            if isinstance(entry, Mapping) and entry.get("group") == "hand" and entry.get("side") == side:
                joint = entry.get("joint")
                if not isinstance(joint, str):
                    raise ValueError(f"manifest {side} hand entry has no joint name")
                ordered.append(joint)
        if len(ordered) != 20 or set(ordered) != set(names):
            raise ValueError(f"manifest {side} actuator order disagrees with hand order")
        return ordered

    return document, side_order("left"), side_order("right")


def _reset_filter(retargeter: Any) -> None:
    reset = getattr(retargeter, "reset_filter", None)
    if reset is None:
        reset = getattr(retargeter, "reset", None)
    if reset is not None:
        reset()


def _frame(value: Any) -> PicoHandFrame:
    if isinstance(value, PicoHandFrame):
        return value
    if isinstance(value, PicoHandsInput):
        return value.frame
    # Convert the transport PicoFrame without importing Zenoh/wire here.
    if hasattr(value, "hands"):
        hands = np.asarray(value.hands)
        return PicoHandFrame(
            hands[0],
            hands[1],
            bool(np.asarray(value.valid)[0]),
            bool(np.asarray(value.valid)[1]),
            int(value.tracking_epoch),
            int(value.sequence_id),
            int(value.timestamp_ns),
            float(np.asarray(value.hand_scale)[0]),
            float(np.asarray(value.hand_scale)[1]),
        )
    return PicoHandsInput(value).frame


class WujiRetargetPair:
    """Retarget both hands while retaining the last safe output per side."""

    def __init__(
        self,
        left: Any,
        right: Any,
        *,
        left_actuator_joint_names: Sequence[str] | None = None,
        right_actuator_joint_names: Sequence[str] | None = None,
        retargeter_factory: Callable[..., Any] | None = None,
    ) -> None:
        if retargeter_factory is None:
            if Retargeter is None:
                raise ImportError("wuji_retargeting is required for WujiRetargetPair")
            retargeter_factory = Retargeter.from_yaml
        self.left_retargeter = self._make(left, "left", retargeter_factory)
        self.right_retargeter = self._make(right, "right", retargeter_factory)
        left_source, right_source = _source_names(self.left_retargeter, "left"), _source_names(
            self.right_retargeter, "right"
        )
        self._left_perm = qpos_reorder_perm(
            left_source, list(left_actuator_joint_names or LEFT_HAND_JOINTS)
        )
        self._right_perm = qpos_reorder_perm(
            right_source, list(right_actuator_joint_names or RIGHT_HAND_JOINTS)
        )
        self._left_limits = _source_limits(self.left_retargeter)
        self._right_limits = _source_limits(self.right_retargeter)
        self._left_target = self._initial_target(self._left_limits, self._left_perm)
        self._right_target = self._initial_target(self._right_limits, self._right_perm)
        self._epoch: int | None = None
        self._last_sequence: int | None = None

    @staticmethod
    def _initial_target(limits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
        if np.all(np.isfinite(limits)):
            return limits.mean(axis=1)[permutation]
        return np.zeros(20, dtype=np.float64)

    @staticmethod
    def _make(value: Any, side: str, factory: Callable[..., Any]) -> Any:
        if hasattr(value, "retarget"):
            return value
        if isinstance(value, Mapping) and Retargeter is not None:
            config = deepcopy(dict(value))
            optimizer = config.setdefault("optimizer", {})
            if not isinstance(optimizer, dict):
                raise ValueError("retarget optimizer config must be a mapping")
            optimizer.setdefault("hand_side", side)
            from_config = getattr(Retargeter, "from_config", None)
            # A mapping has no filesystem path for ``from_yaml``; use the
            # vendor's config constructor whenever it is available.  Custom
            # factories still receive non-mapping values unchanged.
            if from_config is not None:
                return from_config(config, hand_side=side)
        try:
            return factory(value, hand_side=side)
        except TypeError:
            return factory(value, side)

    @classmethod
    def from_yaml(
        cls,
        left_config: str | Path,
        right_config: str | Path,
        *,
        left_actuator_joint_names: Sequence[str] | None = None,
        right_actuator_joint_names: Sequence[str] | None = None,
    ) -> "WujiRetargetPair":
        return cls(
            left_config,
            right_config,
            left_actuator_joint_names=left_actuator_joint_names,
            right_actuator_joint_names=right_actuator_joint_names,
        )

    @classmethod
    def from_manifest(
        cls,
        left_config: str | Path | dict[str, Any],
        right_config: str | Path | dict[str, Any],
        manifest_path: str | Path,
        urdf_path: str | Path,
    ) -> "WujiRetargetPair":
        """Construct the pair using the compiler manifest's hand order.

        Every source hash and joint name is checked before the vendor
        optimizer is created.  This makes a changed URDF or reordered actuator
        list fail closed instead of silently applying a wrong hand pose.
        """

        _, left_names, right_names = _manifest_hand_contract(manifest_path, urdf_path)
        left = _config_from(left_config)
        right = _config_from(right_config)
        for config, side, names in ((left, "left", left_names), (right, "right", right_names)):
            if not isinstance(config, dict):
                # YAML paths are loaded by the vendor factory; inject the
                # strict manifest names only when a mapping was supplied.
                continue
            optimizer = config.setdefault("optimizer", {})
            if not isinstance(optimizer, dict):
                raise ValueError("retarget optimizer config must be a mapping")
            optimizer["hand_side"] = side
            optimizer["urdf_path"] = str(Path(urdf_path).resolve())
            optimizer["active_joint_names"] = list(names)
            optimizer.pop("mjcf_path", None)
        if Retargeter is None:
            raise ImportError("wuji_retargeting is required for WujiRetargetPair")

        def factory(config: dict[str, Any], *, hand_side: str) -> Any:
            return Retargeter.from_config(config, hand_side=hand_side)

        # For YAML paths, load first so manifest names/URDF are injected too.
        if not isinstance(left, dict) or not isinstance(right, dict):
            configs: list[dict[str, Any]] = []
            for value, side, names in ((left_config, "left", left_names), (right_config, "right", right_names)):
                path = Path(value).resolve()
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    raise ValueError(f"retarget config must be a mapping: {path}")
                config["__yaml_dir"] = str(path.parent)
                optimizer = config.setdefault("optimizer", {})
                if not isinstance(optimizer, dict):
                    raise ValueError("retarget optimizer config must be a mapping")
                optimizer.update({
                    "hand_side": side,
                    "urdf_path": str(Path(urdf_path).resolve()),
                    "active_joint_names": list(names),
                })
                optimizer.pop("mjcf_path", None)
                configs.append(config)
            left, right = configs
        return cls(
            left,
            right,
            left_actuator_joint_names=left_names,
            right_actuator_joint_names=right_names,
            retargeter_factory=factory,
        )

    def reset_filter(self, tracking_epoch: int | None = None) -> None:
        _reset_filter(self.left_retargeter)
        _reset_filter(self.right_retargeter)
        self._epoch = None if tracking_epoch is None else int(tracking_epoch)
        self._last_sequence = None

    @staticmethod
    def _mapped(
        retargeter: Any,
        points: np.ndarray,
        permutation: np.ndarray,
        limits: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(retargeter.retarget(points), dtype=np.float64)
        if value.shape != (20,) or not np.all(np.isfinite(value)):
            raise ValueError("retargeter returned an invalid 20-D vector")
        return np.clip(value, limits[:, 0], limits[:, 1])[permutation]

    @staticmethod
    def _resilient_input(frame: PicoHandFrame | Mapping[str, Any] | Any) -> PicoHandsInput:
        """Validate malformed hand arrays/scales independently before solving."""
        safe_hand = np.zeros((26, 7), dtype=np.float64)
        safe_hand[:, 6] = 1.0

        def safe_flag(value: Any, default: bool = False) -> tuple[bool, bool]:
            if type(value) is bool:
                return value, True
            if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                return bool(value), True
            return default, False

        def safe_counter(
            value: Any,
            default: int = 0,
            *,
            upper: int = 0xFFFFFFFFFFFFFFFF,
        ) -> tuple[int, bool]:
            if isinstance(value, bool) or not isinstance(value, Integral):
                return default, False
            value = int(value)
            return (value, True) if 0 <= value <= upper else (default, False)

        if not isinstance(frame, (PicoHandFrame, Mapping)):
            # TrackingFrame-like objects normally go through PicoHandsInput;
            # if one side or a metadata field is malformed, fall back to the
            # same per-side sanitization used for mapping inputs instead of
            # allowing a callback exception to tear down the viewer loop.
            try:
                return PicoHandsInput(frame)
            except Exception:
                left = getattr(frame, "left_hand", safe_hand)
                right = getattr(frame, "right_hand", safe_hand)
                left_active, left_flag_ok = safe_flag(
                    getattr(frame, "left_active", True), True
                )
                right_active, right_flag_ok = safe_flag(
                    getattr(frame, "right_active", True), True
                )
                tracking_epoch, epoch_ok = safe_counter(
                    getattr(frame, "tracking_epoch", 0)
                )
                sequence_id, sequence_ok = safe_counter(
                    getattr(frame, "sequence_id", 0)
                )
                timestamp_ns, timestamp_ok = safe_counter(
                    getattr(frame, "timestamp_ns", 0),
                    upper=0x7FFFFFFFFFFFFFFF,
                )
                metadata_ok = epoch_ok and sequence_ok and timestamp_ok
                frame = {
                    "left_hand": left,
                    "right_hand": right,
                    "left_active": left_active if left_flag_ok and metadata_ok else False,
                    "right_active": right_active if right_flag_ok and metadata_ok else False,
                    "tracking_epoch": tracking_epoch,
                    "sequence_id": sequence_id,
                    "timestamp_ns": timestamp_ns,
                    "left_scale": getattr(frame, "left_scale", 1.0),
                    "right_scale": getattr(frame, "right_scale", 1.0),
                }
        if isinstance(frame, PicoHandFrame):
            raw = frame
        else:
            left_present = "left_hand" in frame
            right_present = "right_hand" in frame
            left_active, left_flag_ok = safe_flag(
                frame.get("left_active", True), True
            )
            right_active, right_flag_ok = safe_flag(
                frame.get("right_active", True), True
            )
            tracking_epoch, epoch_ok = safe_counter(frame.get("tracking_epoch", 0))
            sequence_id, sequence_ok = safe_counter(frame.get("sequence_id", 0))
            timestamp_ns, timestamp_ok = safe_counter(
                frame.get("timestamp_ns", 0), upper=0x7FFFFFFFFFFFFFFF
            )
            metadata_ok = epoch_ok and sequence_ok and timestamp_ok
            raw = PicoHandFrame(
                frame.get("left_hand", safe_hand),
                frame.get("right_hand", safe_hand),
                left_present and left_flag_ok and left_active and metadata_ok,
                right_present and right_flag_ok and right_active and metadata_ok,
                tracking_epoch,
                sequence_id,
                timestamp_ns,
                frame.get("left_scale", 1.0),
                frame.get("right_scale", 1.0),
            )

        def checked(value: Any, name: str) -> tuple[np.ndarray, bool]:
            try:
                return PicoHandsInput._array(value, name), True
            except (TypeError, ValueError):
                safe = np.zeros((26, 7), dtype=np.float64)
                safe[:, 6] = 1.0
                return safe, False

        left, left_ok = checked(raw.left_hand, "left_hand")
        right, right_ok = checked(raw.right_hand, "right_hand")
        try:
            left_scale = PicoHandsInput._scale(raw.left_scale, "left_scale")
            left_scale_ok = True
        except (TypeError, ValueError):
            left_scale, left_scale_ok = 1.0, False
        try:
            right_scale = PicoHandsInput._scale(raw.right_scale, "right_scale")
            right_scale_ok = True
        except (TypeError, ValueError):
            right_scale, right_scale_ok = 1.0, False
        left_active, left_active_ok = safe_flag(raw.left_active, False)
        right_active, right_active_ok = safe_flag(raw.right_active, False)
        tracking_epoch, epoch_ok = safe_counter(getattr(raw, "tracking_epoch", 0))
        sequence_id, sequence_ok = safe_counter(getattr(raw, "sequence_id", 0))
        timestamp_ns, timestamp_ok = safe_counter(
            getattr(raw, "timestamp_ns", 0),
            upper=0x7FFFFFFFFFFFFFFF,
        )
        metadata_ok = epoch_ok and sequence_ok and timestamp_ok
        safe = PicoHandFrame(
            left,
            right,
            bool(
                left_active
                and left_active_ok
                and left_ok
                and left_scale_ok
                and metadata_ok
            ),
            bool(
                right_active
                and right_active_ok
                and right_ok
                and right_scale_ok
                and metadata_ok
            ),
            tracking_epoch,
            sequence_id,
            timestamp_ns,
            left_scale,
            right_scale,
        )
        return PicoHandsInput(safe)

    def _held(self, frame: PicoHandFrame, reason: HandHoldReason) -> RetargetedHands:
        return RetargetedHands(
            frame.tracking_epoch,
            frame.sequence_id,
            self._left_target,
            self._right_target,
            False,
            False,
            reason,
            reason,
        )

    def retarget(self, frame: Any) -> RetargetedHands:
        input_frame = self._resilient_input(frame)
        source = input_frame.frame
        if self._epoch != source.tracking_epoch:
            self.reset_filter(source.tracking_epoch)
        elif self._last_sequence is not None and source.sequence_id <= self._last_sequence:
            return self._held(source, HandHoldReason.STALE)
        self._last_sequence = source.sequence_id
        left_valid = right_valid = False
        left_reason, right_reason = HandHoldReason.INACTIVE, HandHoldReason.INACTIVE
        if source.left_active:
            try:
                self._left_target = self._mapped(
                    self.left_retargeter,
                    input_frame.get_side_raw_fingers_data("left"),
                    self._left_perm,
                    self._left_limits,
                )
                left_valid, left_reason = True, HandHoldReason.NONE
            except Exception:
                left_reason = HandHoldReason.SOLVER_FAILURE
        if source.right_active:
            try:
                self._right_target = self._mapped(
                    self.right_retargeter,
                    input_frame.get_side_raw_fingers_data("right"),
                    self._right_perm,
                    self._right_limits,
                )
                right_valid, right_reason = True, HandHoldReason.NONE
            except Exception:
                right_reason = HandHoldReason.SOLVER_FAILURE
        return RetargetedHands(
            source.tracking_epoch,
            source.sequence_id,
            self._left_target,
            self._right_target,
            left_valid,
            right_valid,
            left_reason,
            right_reason,
        )


__all__ = [
    "HandHoldReason",
    "RetargetedHands",
    "WujiRetargetPair",
    "mjcf_actuator_joint_names",
    "PICO_TO_MEDIAPIPE",
    "qpos_reorder_perm",
]
