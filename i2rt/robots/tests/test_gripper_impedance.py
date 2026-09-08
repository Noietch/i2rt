"""Exercise the real command-update path without constructing hardware or threads."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import JointMapper


def test_eva_factory_wires_impedance_to_sdk_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.hardware.yam.robot import _sdk_factory

    from i2rt.robots import get_robot

    chain = Mock()
    chain.motor_offset = np.zeros(7)
    chain.read_states.return_value = [SimpleNamespace(pos=0.0) for _ in range(7)]
    monkeypatch.setattr(get_robot, "DMChainCanInterface", Mock(return_value=chain))
    monkeypatch.setattr(get_robot, "combine_arm_and_gripper_xml", Mock(return_value="unused.xml"))
    monkeypatch.setattr(get_robot, "_load_joint_limits_from_xml", Mock(return_value=np.array([[-3.0, 3.0]] * 7)))
    constructor = Mock()
    monkeypatch.setattr(get_robot, "MotorChainRobot", constructor)

    result = _sdk_factory(
        channel="unused_can",
        arm_type="yam",
        gripper_type="linear_4310",
        enable_auto_recovery=False,
        gripper_limits_override=np.array([0.0, 6.57]),
        gripper_kp=3.0,
        gripper_damping=0.6,
        gripper_close_torque_limit=0.2,
        gripper_open_torque_limit=0.4,
    )

    assert result is constructor.return_value
    kwargs = constructor.call_args.kwargs
    assert kwargs["gripper_torque_limits"] == (0.4, 0.2)
    assert kwargs["kp"][6] == 3.0
    assert kwargs["kd"][6] == 0.6
    assert kwargs["limit_gripper_force"] == -1.0
    assert kwargs["enable_gripper_calibration"] is False


@pytest.fixture(params=[(0.0, 6.57), (6.57, 0.0)])
def robot(request: pytest.FixtureRequest) -> MotorChainRobot:
    arm = MotorChainRobot.__new__(MotorChainRobot)
    arm.motor_chain = [None] * 7
    arm._command_lock = threading.Lock()
    arm._state_lock = threading.Lock()
    arm._gripper_index = 6
    arm._gripper_limits = np.array(request.param)
    arm.remapper = JointMapper({6: arm._gripper_limits}, 7)
    arm._joint_limits = None
    arm._joint_state = SimpleNamespace(pos=np.array([0.0] * 6 + [0.5]), vel=np.zeros(7))
    arm.use_gravity_comp = False
    arm.use_coulomb_friction = False
    arm.gravity_comp_factor = np.ones(7)
    arm._clip_motor_torque = np.inf
    arm._gripper_torque_limits = (0.4, 0.29)
    arm._limit_gripper_force = 50.0
    arm._gripper_force_limiter = Mock()
    arm._kp = np.array([80.0] * 6 + [5.0])
    arm._kd = np.array([1.5] * 6 + [0.5])
    arm._grav_comp_kd = np.array([1.5] * 6 + [0.0])
    arm._update_joint_state = Mock()
    arm.command_joint_pos(arm._joint_state.pos.copy())
    return arm


@pytest.mark.parametrize("target,expected", [(0.0, 0.29), (1.0, -0.4)])
def test_complete_output_is_bounded_before_contact(robot: MotorChainRobot, target: float, expected: float) -> None:
    robot.command_joint_pos(np.array([0.1] * 6 + [target]))
    closing_sign = np.sign(robot._gripper_limits[0] - robot._gripper_limits[1])
    robot._commands.torques[:6] = 0.2
    for _ in range(50):
        robot.update()
        torque, commands = robot._update_joint_state.call_args.args
        assert torque[6] * closing_sign == pytest.approx(expected)
        assert commands.kp[6] == commands.kd[6] == 0.0
        np.testing.assert_allclose(torque[:6], 0.2)
        np.testing.assert_allclose(commands.kp[:6], 80.0)
        np.testing.assert_allclose(commands.kd[:6], 1.5)
        np.testing.assert_allclose(commands.pos[:6], 0.1)
        assert robot._commands.kp[6] == 5.0
    robot._gripper_force_limiter.update.assert_not_called()
    np.testing.assert_allclose(robot.get_motor_torques(), torque)


def test_spring_and_damping_use_motor_units(robot: MotorChainRobot) -> None:
    robot.command_joint_pos(np.array([0.0] * 6 + [0.499]))
    robot._joint_state.vel[6] = -0.01
    span = robot._gripper_limits[1] - robot._gripper_limits[0]
    expected = 5.0 * (-0.001 * span) - 0.5 * (-0.01 * span)
    robot.update()
    torque, _ = robot._update_joint_state.call_args.args
    assert torque[6] == pytest.approx(expected)


def test_damping_and_feedforward_are_inside_limit(robot: MotorChainRobot) -> None:
    closing_sign = np.sign(robot._gripper_limits[0] - robot._gripper_limits[1])
    # At zero position error, opening motion alone requests a large closing brake.
    robot._joint_state.vel[6] = 10.0
    robot._commands.torques[6] = closing_sign * 2.0
    robot.update()
    torque, commands = robot._update_joint_state.call_args.args
    assert torque[6] * closing_sign == pytest.approx(0.29)
    assert commands.kd[6] == 0.0


def test_release_reverses_torque_without_latch(robot: MotorChainRobot) -> None:
    closing_sign = np.sign(robot._gripper_limits[0] - robot._gripper_limits[1])
    robot.command_joint_pos(np.zeros(7))
    robot.update()
    robot.command_joint_pos(np.array([0.0] * 6 + [1.0]))
    robot.update()
    torque, _ = robot._update_joint_state.call_args.args
    assert torque[6] * closing_sign == pytest.approx(-0.4)


def test_gravity_idle_removes_gripper_output(robot: MotorChainRobot) -> None:
    robot.command_joint_pos(np.zeros(7))
    robot.update()
    robot.enter_gravity_comp_idle()
    robot._joint_state.vel[6] = 0.1
    robot.update()
    torque, commands = robot._update_joint_state.call_args.args
    assert torque[6] == 0.0
    assert commands.kp[6] == commands.kd[6] == 0.0


def test_global_torque_clip_remains_effective(robot: MotorChainRobot) -> None:
    robot._clip_motor_torque = 0.1
    robot.command_joint_pos(np.zeros(7))
    robot.update()
    torque, _ = robot._update_joint_state.call_args.args
    assert abs(torque[6]) == pytest.approx(0.1)


def test_invalid_feedback_sends_finite_zero_gripper_torque(robot: MotorChainRobot) -> None:
    robot._joint_state.pos[6] = np.nan
    robot.update()
    torque, commands = robot._update_joint_state.call_args.args
    assert torque[6] == 0.0
    assert np.isfinite(commands.pos[6])
    assert commands.kp[6] == commands.kd[6] == 0.0


def test_sdk_default_retains_onboard_position_control(robot: MotorChainRobot) -> None:
    robot._gripper_torque_limits = None
    robot._limit_gripper_force = -1
    robot.command_joint_pos(np.zeros(7))
    robot.update()
    torque, commands = robot._update_joint_state.call_args.args
    assert torque[6] == 0.0
    assert commands.kp[6] == 5.0
    assert commands.kd[6] == 0.5
