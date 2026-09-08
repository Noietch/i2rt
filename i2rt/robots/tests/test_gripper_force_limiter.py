import time

import pytest

from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType


def test_linear_gripper_force_limit_does_not_add_extra_torque() -> None:
    limiter = GripperForceLimiter(
        max_force=20.0,
        gripper_type=GripperType.LINEAR_4310,
        arm_type=ArmType.YAM,
        kp=5.0,
        effort_compensation=0.0,
    )
    limiter._past_gripper_effort_buffer.put(time.time(), 0.6)

    torque = limiter.compute_target_gripper_torque(
        {
            "current_qvel": 0.0,
            "current_normalized_qpos": 0.5,
            "target_normalized_qpos": 0.0,
            "current_qpos": 0.0,
        }
    )

    assert torque == pytest.approx(20.0 * 0.096 / 6.57)
