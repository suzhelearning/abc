from typing import Iterable, List
import time

import numpy as np
import numpy.typing as npt
import pinocchio as pin

class RobotWrapper:
    """Pinocchio robot wrapper for a full or side-specific reduced URDF model."""

    def __init__(
        self,
        urdf_path: str,
        hand_side: str | None = None,
        active_joint_names: Iterable[str] | None = None,
    ):
        full_model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        if full_model.nv != full_model.nq:
            raise NotImplementedError("Cannot handle robot with special joint.")
        self.hand_side = hand_side.lower() if hand_side else None
        if self.hand_side not in {None, "left", "right"}:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

        all_dof_names = [
            name for index, name in enumerate(full_model.names) if full_model.nqs[index] > 0
        ]
        if active_joint_names is None:
            active = all_dof_names
        else:
            active = list(active_joint_names)
            if len(active) != len(set(active)):
                raise ValueError("active_joint_names must be unique")
            missing = [name for name in active if name not in all_dof_names]
            if missing:
                raise ValueError(f"active joints are absent or fixed in URDF: {missing}")
            if self.hand_side is not None:
                prefix = f"{self.hand_side[0]}_"
                wrong_side = [
                    name for name in active
                    if name.startswith(("l_", "r_")) and not name.startswith(prefix)
                ]
                if wrong_side:
                    raise ValueError(
                        f"active joints contain the other hand for {self.hand_side}: {wrong_side}"
                    )
                if any(name.startswith("Joint") for name in active):
                    raise ValueError("arm joints cannot be active in a reduced hand model")
            if len(active) != 20:
                raise ValueError(
                    f"reduced hand model requires exactly 20 active joints, got {len(active)}"
                )

        active_set = set(active)
        inactive_ids = [
            joint_id
            for joint_id, name in enumerate(full_model.names)
            if full_model.nqs[joint_id] > 0 and name not in active_set
        ]
        # The authoritative URDF contains duplicate fixed/body marker frames;
        # Pinocchio's reducer requires unique names although hand frames remain unchanged.
        seen_frames: set[str] = set()
        for frame_id, frame in enumerate(full_model.frames):
            if frame.name in seen_frames:
                frame.name = f"{frame.name}__duplicate_{frame_id}"
            seen_frames.add(frame.name)
        reference = pin.neutral(full_model)
        if active_joint_names is not None:
            try:
                self.model = pin.buildReducedModel(full_model, [0, *inactive_ids], reference)
            except ValueError as exc:
                if "parent joint is not valid" not in str(exc):
                    raise
                # Some Pinocchio versions reject universe in this malformed
                # fixed-frame tree; it has no DoF, so omitting it is equivalent.
                self.model = pin.buildReducedModel(full_model, inactive_ids, reference)
        else:
            self.model = full_model
        self.data: pin.Data = self.model.createData()
        self.active_joint_names = tuple(active)

        # Timing statistics for FK and Jacobian
        self._timing_enabled = False
        self._fk_time_sum = 0.0
        self._jacobian_time_sum = 0.0
        self._fk_call_count = 0
        self._jacobian_call_count = 0

    def enable_timing(self, enabled: bool = True):
        """Enable or disable timing statistics."""
        self._timing_enabled = enabled
        if enabled:
            self.reset_timing()

    def reset_timing(self):
        """Reset timing statistics."""
        self._fk_time_sum = 0.0
        self._jacobian_time_sum = 0.0
        self._fk_call_count = 0
        self._jacobian_call_count = 0

    def get_timing_stats(self):
        """Get timing statistics.

        Returns:
            dict with keys:
                - fk_avg_us: average FK time in microseconds
                - jacobian_avg_us: average Jacobian time in microseconds
                - fk_call_count: number of FK calls
                - jacobian_call_count: number of Jacobian calls
        """
        fk_avg = (self._fk_time_sum / self._fk_call_count * 1e6) if self._fk_call_count > 0 else 0
        jac_avg = (self._jacobian_time_sum / self._jacobian_call_count * 1e6) if self._jacobian_call_count > 0 else 0
        return {
            'fk_avg_us': fk_avg,
            'jacobian_avg_us': jac_avg,
            'fk_call_count': self._fk_call_count,
            'jacobian_call_count': self._jacobian_call_count,
        }

    @property
    def dof_joint_names(self) -> List[str]:
        """Return names of joints with DOF > 0."""
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def joint_limits(self):
        """Return joint limits as (lower, upper) pairs."""
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    def get_link_index(self, name: str) -> int:
        """Return a frame ID while preventing cross-hand frame resolution."""
        if self.hand_side and name.startswith(("l_", "r_")):
            expected = f"{self.hand_side[0]}_"
            if not name.startswith(expected):
                raise RuntimeError(f"frame '{name}' belongs to the other hand")
        candidates = [name]
        if self.hand_side:
            candidates.append(f"{self.hand_side}_{name}")
        for candidate in candidates:
            idx = self.model.getFrameId(candidate, pin.BODY)
            if idx < self.model.nframes:
                return idx
        raise RuntimeError(
            f"Frame '{name}' not found. "
            f"Available: {[self.model.frames[i].name for i in range(self.model.nframes)]}"
        )

    def get_actuated_qpos_index(self, link_name: str) -> int:
        """Return the qpos index of the joint that actuates ``link_name``.

        A link's parent joint is the joint directly above it in the kinematic
        chain (the joint whose URDF ``<child>`` is this link). For the wuji
        finger chain ``link1->link2->link3->link4``, ``finger{i}_link3``'s parent
        joint is the PIP joint and ``finger{i}_link4``'s is the DIP joint.

        Resolving the qpos slot this way (instead of hardcoding indices) keeps the
        mapping correct when a custom URDF declares joints in a different order or
        with a non-uniform DOF layout. Raises if the link is unknown or its parent
        joint is not a single-DOF joint.
        """
        frame_id = self.get_link_index(link_name)
        joint_id = self.model.frames[frame_id].parentJoint
        joint = self.model.joints[joint_id]
        if joint.nq != 1:
            raise RuntimeError(
                f"Parent joint of '{link_name}' ('{self.model.names[joint_id]}') has "
                f"nq={joint.nq}, expected a single-DOF (revolute/prismatic) joint."
            )
        return int(joint.idx_q)

    def compute_forward_kinematics(self, qpos: npt.NDArray):
        """Compute forward kinematics for all links."""
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        """Get link pose as 4x4 homogeneous matrix."""
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        """Compute Jacobian for a single link."""
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J

    def compute_all_jacobians_batch(self, qpos: npt.NDArray, link_indices: List[int]) -> npt.NDArray:
        """Batch compute position Jacobians for multiple links.

        This is more efficient than calling compute_single_link_local_jacobian
        multiple times because it uses computeJointJacobians once.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices

        Returns:
            jacobians: (num_links, 3, nq) position Jacobians in world frame
        """
        qpos = np.asarray(qpos, dtype=np.float64)

        # Compute all joint Jacobians at once (updates data.J internally)
        pin.computeJointJacobians(self.model, self.data, qpos)
        # Update all frame placements
        pin.updateFramePlacements(self.model, self.data)

        jacobians = []
        for idx in link_indices:
            # getFrameJacobian reuses computed joint Jacobians (faster than computeFrameJacobian)
            J_local = pin.getFrameJacobian(self.model, self.data, idx, pin.LOCAL)
            # Get rotation to transform to world frame
            R = self.data.oMf[idx].rotation
            # Only take position part (3, nq) and transform to world frame
            J_world_pos = R @ J_local[:3, :]
            jacobians.append(J_world_pos)

        return np.stack(jacobians, axis=0)

    def compute_fk_batch(self, qpos: npt.NDArray, link_indices: List[int]) -> npt.NDArray:
        """Batch compute FK positions for multiple links.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices

        Returns:
            positions: (num_links * 3,) flattened positions
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)

        positions = []
        for idx in link_indices:
            pos = self.data.oMf[idx].translation
            positions.append(pos)

        return np.concatenate(positions)
