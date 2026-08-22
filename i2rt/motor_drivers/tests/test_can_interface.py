import can

from i2rt.motor_drivers.can_interface import CanInterface


class _FakeBus:
    channel_info = "fake-can"

    def __init__(self, messages: list[can.Message]) -> None:
        self.messages = list(messages)
        self.sent: list[can.Message] = []

    def send(self, message: can.Message) -> None:
        self.sent.append(message)

    def recv(self, timeout: float) -> can.Message | None:
        del timeout
        return self.messages.pop(0) if self.messages else None


def test_request_waits_for_matching_response_after_unrelated_frame() -> None:
    unrelated = can.Message(arbitration_id=0x12, data=bytes(8))
    expected = can.Message(arbitration_id=0x11, data=bytes(8))
    interface = object.__new__(CanInterface)
    interface.bus = _FakeBus([unrelated, expected])
    interface.name = "yam_real"
    interface.channel = "can2"
    interface.use_buffered_reader = False

    response = interface._send_message_get_response(
        id=1,
        motor_id=1,
        data=[0] * 8,
        max_retry=1,
        expected_id=0x11,
    )

    assert response is expected
    assert len(interface.bus.sent) == 1


def test_request_suppresses_timeout_warning_until_retries_are_exhausted() -> None:
    expected = can.Message(arbitration_id=0x11, data=bytes(8))
    interface = object.__new__(CanInterface)
    interface.bus = _FakeBus([])
    interface.name = "yam_real"
    interface.channel = "can2"
    interface.receive_mode = type("_Mode", (), {"get_receive_id": lambda self, _id: 0x11})()
    calls: list[bool] = []

    def receive(*_args: object, supress_warning: bool = False, **_kwargs: object) -> can.Message | None:
        calls.append(supress_warning)
        return expected if len(calls) == 2 else None

    interface._receive_message = receive

    response = interface._send_message_get_response(
        id=1,
        motor_id=1,
        data=[0] * 8,
        max_retry=2,
    )

    assert response is expected
    assert calls == [True, True]
