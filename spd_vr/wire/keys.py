"""Stable Zenoh keys for the SPD VR teleoperation session."""

TRACKING_KEY = "spd/vr/v1/tracking"
ARM_TARGETS_KEY = "spd/vr/v1/arm_targets"
CONTROL_KEY = "spd/vr/v1/control"
STATUS_BRIDGE_KEY = "spd/vr/v1/status/bridge"
STATUS_IK_KEY = "spd/vr/v1/status/ik"
STATUS_VIEWER_KEY = "spd/vr/v1/status/viewer"

__all__ = [
    "ARM_TARGETS_KEY",
    "CONTROL_KEY",
    "STATUS_BRIDGE_KEY",
    "STATUS_IK_KEY",
    "STATUS_VIEWER_KEY",
    "TRACKING_KEY",
]
