import struct

import can
import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import PassiveEncoderReader


@pytest.mark.parametrize(
    ("encoded_position", "expected_ticks"),
    [
        (0x0014, 20),
        (0x0FFC, -4),
        (0x0E32, -462),
        (0xFFFC, -4),
    ],
)
def test_passive_encoder_sign_extends_12_bit_position(
    encoded_position: int,
    expected_ticks: int,
) -> None:
    message = can.Message(
        arbitration_id=0x50F,
        data=struct.pack("!B H h B", 0, encoded_position, -7, 0b11),
    )

    position, velocity, buttons = PassiveEncoderReader._parse_encoder_message(
        object.__new__(PassiveEncoderReader),
        message,
    )

    radians_per_tick = 2 * np.pi / 4096
    assert position == pytest.approx(expected_ticks * radians_per_tick)
    assert velocity == pytest.approx(-7 * radians_per_tick)
    assert buttons == [1, 1]
