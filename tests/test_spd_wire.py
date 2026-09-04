from dataclasses import replace

import numpy as np
import pytest

from spd_vr.wire import (
    ARM_TARGET_PACKET_SIZE,
    ArmTargetFrame,
    ArmTargetHoldReason,
    ArmTargetProtocolError,
    ArmTargetStreamDecoder,
    TRACKING_PACKET_SIZE,
    TrackingFrame,
    TrackingProtocolError,
    decode_tracking,
    encode_tracking,
    decode_arm_target,
    encode_arm_target,
)


def _poses(shape):
    value = np.zeros(shape, dtype=np.float32)
    value[..., 6] = 1.0
    return value


def test_tracking_packet_round_trip_and_crc():
    frame = TrackingFrame(
        sequence=1,
        tracking_epoch=1,
        source_timestamp_ns=10,
        bridge_monotonic_ns=20,
        left_active=True,
        right_active=True,
        head_valid=True,
        left_scale=1.0,
        right_scale=1.0,
        head_pose=_poses((1, 7))[0],
        left_hand=_poses((26, 7)),
        right_hand=_poses((26, 7)),
    )
    packet = encode_tracking(frame)
    assert len(packet) == TRACKING_PACKET_SIZE
    decoded = decode_tracking(packet)
    np.testing.assert_array_equal(decoded.left_hand, frame.left_hand)

    corrupted = bytearray(packet)
    corrupted[-1] ^= 1
    try:
        decode_tracking(corrupted)
    except TrackingProtocolError as exc:
        assert exc.code == "crc_mismatch"
    else:
        raise AssertionError("corrupt packet was accepted")


def _arm_frame(sequence=1, epoch=2):
    return ArmTargetFrame(
        sequence=sequence,
        tracking_epoch=epoch,
        source_timestamp_ns=10,
        control_timestamp_ns=20,
        valid_mask=3,
        left_hold_reason=ArmTargetHoldReason.NONE,
        right_hold_reason=ArmTargetHoldReason.NONE,
        left_q=tuple(float(i) for i in range(7)),
        right_q=tuple(float(i + 7) for i in range(7)),
        left_qdot=tuple(0.1 for _ in range(7)),
        right_qdot=tuple(-0.1 for _ in range(7)),
    )


def test_arm_target_packet_round_trip_size_crc_and_epoch_gate():
    frame = _arm_frame()
    packet = encode_arm_target(frame)
    assert len(packet) == ARM_TARGET_PACKET_SIZE == 272
    decoded = decode_arm_target(packet)
    assert decoded == frame
    stream = ArmTargetStreamDecoder()
    assert stream.decode(packet).sequence == 1
    with pytest.raises(ArmTargetProtocolError, match="out_of_order"):
        stream.decode(packet)
    stale = decode_arm_target(packet, now_ns=100, max_age_ns=1)
    assert stale.valid_mask == 0
    assert stale.left_hold_reason is ArmTargetHoldReason.INPUT_STALE
    future = decode_arm_target(packet, now_ns=19, max_age_ns=1)
    assert future.valid_mask == 0
    assert future.right_hold_reason is ArmTargetHoldReason.INPUT_STALE


def test_arm_target_hold_reason_must_match_validity_and_crc_rejects_tampering():
    with pytest.raises(ArmTargetProtocolError, match="valid_mask"):
        encode_arm_target(replace(_arm_frame(), valid_mask=0))
    packet = bytearray(encode_arm_target(_arm_frame()))
    packet[60] ^= 1
    with pytest.raises(ArmTargetProtocolError, match="crc_mismatch"):
        decode_arm_target(packet)


def test_arm_target_mapping_does_not_truncate_malformed_metadata_or_vectors():
    frame = _arm_frame()
    values = frame.__dict__.copy()
    values["sequence"] = "1"
    with pytest.raises(ArmTargetProtocolError, match="invalid_metadata"):
        encode_arm_target(values)

    values = frame.__dict__.copy()
    values["left_q"] = tuple([0.0] * 6 + ["1.5"])
    with pytest.raises(ArmTargetProtocolError, match="invalid_vector"):
        encode_arm_target(values)

    with pytest.raises(ArmTargetProtocolError, match="invalid_metadata"):
        decode_arm_target(encode_arm_target(frame), last_sequence="1")
    with pytest.raises(ArmTargetProtocolError, match="invalid_metadata"):
        decode_arm_target(encode_arm_target(frame), max_age_ns=1.5)
    with pytest.raises(ArmTargetProtocolError, match="invalid_metadata"):
        ArmTargetStreamDecoder(max_age_ns="50000000")
