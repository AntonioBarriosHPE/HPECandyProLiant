import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp.test_utils import TestClient, TestServer

import app
import hardware


class SerialDiscoveryTests(unittest.TestCase):
    def test_discovery_rejects_open_but_unrelated_port_and_detects_baud(self):
        ports = [SimpleNamespace(device="COM3"), SimpleNamespace(device="COM8")]
        unrelated = Mock()
        controller = Mock()
        controller.readline.return_value = b"ACK SPIN set to 0\r\n"

        def open_serial(port, baudrate, timeout):
            if port == "COM8" and baudrate == 115200:
                return controller
            return unrelated

        with patch.object(hardware.serial.tools.list_ports, "comports", return_value=ports), patch.object(
            hardware.serial, "Serial", side_effect=open_serial
        ), patch.object(hardware.ArduinoMotorController, "_verify_serial_controller", side_effect=lambda conn: conn is controller):
            self.assertEqual(hardware.ArduinoMotorController.get_auto_detect_connection(), ("COM8", 115200))
        unrelated.close.assert_called()
        controller.close.assert_called_once()

    def test_no_ports_or_busy_ports_do_not_select_a_controller(self):
        for ports in ([], [SimpleNamespace(device="COM9")]):
            with self.subTest(ports=ports), patch.object(
                hardware.serial.tools.list_ports, "comports", return_value=ports
            ), patch.object(hardware.serial, "Serial", side_effect=hardware.serial.SerialException("busy")):
                self.assertIsNone(hardware.ArduinoMotorController.get_auto_detect_connection())

    def test_verification_uses_configuration_only_not_motor_activation(self):
        connection = Mock()
        connection.readline.return_value = b"ACK SPIN set to 0\r\n"
        self.assertTrue(hardware.ArduinoMotorController._verify_serial_controller(connection))
        connection.write.assert_called_once_with(b"SPIN:0\n")

    def test_unrelated_serial_response_is_rejected(self):
        connection = Mock()
        connection.readline.return_value = b"Some other serial device\n"
        with patch.object(hardware.time, "monotonic", side_effect=[0, 0, 5]):
            self.assertFalse(hardware.ArduinoMotorController._verify_serial_controller(connection))

    def test_port_override_is_verified_at_both_baud_rates(self):
        connection = Mock()
        with patch.object(hardware.serial, "Serial", return_value=connection) as open_serial, patch.object(
            hardware.ArduinoMotorController, "_verify_serial_controller", side_effect=[False, True]
        ), patch.object(hardware.serial.tools.list_ports, "comports") as enumerate_ports:
            self.assertEqual(
                hardware.ArduinoMotorController.get_auto_detect_connection(port="COM12"),
                ("COM12", 115200),
            )
        enumerate_ports.assert_not_called()
        self.assertEqual([call.kwargs["baudrate"] for call in open_serial.call_args_list], [9600, 115200])
        self.assertEqual(connection.close.call_count, 2)


class FakeCapture:
    def __init__(self):
        self.sequence = 0
        self.release_count = 0
        self.owner_thread_id = threading.get_ident()
        self.read_thread_ids = set()

    def read(self):
        self.read_thread_ids.add(threading.get_ident())
        time.sleep(0.001)
        self.sequence += 1
        return True, self.sequence

    def release(self):
        self.release_count += 1


class FailingCapture:
    def read(self):
        raise RuntimeError("simulated camera failure")

    def release(self):
        pass


class BlockingCapture:
    def __init__(self):
        self.released = threading.Event()
        self.release_count = 0

    def read(self):
        self.released.wait(timeout=2)
        return False, None

    def release(self):
        self.release_count += 1
        self.released.set()


class CaptureWorkerTests(unittest.TestCase):
    def test_capture_worker_keeps_latest_frame_and_releases_once(self):
        captures = []

        def create_capture():
            capture = FakeCapture()
            captures.append(capture)
            return capture

        worker = app.LatestFrameCapture(create_capture)
        worker.start()
        deadline = time.monotonic() + 1
        sample = None
        while time.monotonic() < deadline:
            sample = worker.read_latest(0)
            if sample and sample[0] > 1:
                break
            time.sleep(0.002)

        worker.stop()

        capture = captures[0]
        self.assertIsNotNone(sample)
        self.assertGreater(sample[0], 1)
        self.assertIn(capture.owner_thread_id, capture.read_thread_ids)
        self.assertEqual(capture.release_count, 1)
        self.assertFalse(worker._thread.is_alive())

    def test_capture_worker_surfaces_reader_exception(self):
        worker = app.LatestFrameCapture(FailingCapture)
        worker.start()
        deadline = time.monotonic() + 1
        while worker.error is None and time.monotonic() < deadline:
            time.sleep(0.002)

        worker.stop()

        self.assertEqual(worker.error, "Camera read raised an exception.")
        self.assertFalse(worker._thread.is_alive())

    def test_stop_unblocks_a_blocked_camera_read(self):
        capture = BlockingCapture()
        worker = app.LatestFrameCapture(lambda: capture)
        worker.start()
        worker.stop()

        self.assertEqual(capture.release_count, 1)
        self.assertFalse(worker._thread.is_alive())


class AsyncStabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_uses_detected_port_and_baud(self):
        for override in (None, "COM12"):
            with self.subTest(override=override), patch.object(
                app, "ARDUINO_SERIAL_PORT", override
            ), patch.object(app, "initialize_openvino_models", return_value=True), patch.object(
                app, "MOTOR_CONTROLLER", None
            ), patch.object(app, "emotion_history"), patch.object(app, "ArduinoMotorController") as controller_class:
                controller_class.get_auto_detect_connection.return_value = ("COM12", 115200)
                motor = controller_class.return_value
                motor.connect = AsyncMock(return_value=True)
                motor.configure_spin_pwm = AsyncMock(return_value=True)
                motor.configure_run_duration = AsyncMock(return_value=True)
                motor.activate_motor = AsyncMock()

                await app.on_aiohttp_startup(None)

                controller_class.get_auto_detect_connection.assert_called_once_with(port=override)
                controller_class.assert_called_once_with(port="COM12", baud_rate=115200)
                motor.activate_motor.assert_not_awaited()

    async def test_startup_without_verified_controller_uses_simulation(self):
        with patch.object(app, "initialize_openvino_models", return_value=True), patch.object(
            app, "MOTOR_CONTROLLER", None
        ), patch.object(app, "emotion_history"), patch.object(app, "ArduinoMotorController") as controller_class:
            controller_class.get_auto_detect_connection.return_value = None
            await app.on_aiohttp_startup(None)
            controller_class.assert_not_called()
            self.assertIsInstance(app.MOTOR_CONTROLLER, hardware.DummyMotorController)
            self.assertFalse(app.MOTOR_CONTROLLER.is_busy())

    async def test_serial_connection_refuses_unverified_device(self):
        controller = hardware.ArduinoMotorController("COM12")
        with patch.object(hardware.serial, "Serial", return_value=Mock()), patch.object(
            hardware.time, "sleep"
        ), patch.object(controller, "_verify_serial_controller", return_value=False), patch.object(
            controller, "disconnect"
        ) as disconnect:
            self.assertFalse(await controller.connect())
            disconnect.assert_called_once()
            self.assertFalse(controller._is_reader_active)

    async def test_websocket_requires_play_and_stops_when_play_owner_disconnects(self):
        producer_stopped = asyncio.Event()

        async def fake_producer():
            try:
                await asyncio.Event().wait()
            finally:
                producer_stopped.set()

        server_app = app.web.Application()
        server_app.router.add_get("/ws", app.websocket_handler)
        with patch.multiple(
            app,
            models_initialized_successfully=True,
            is_demo_active=False,
            demo_owner=None,
            active_websockets=set(),
            video_producer_task=None,
            MOTOR_CONTROLLER=None,
            video_stream_producer_loop=fake_producer,
        ):
            async with TestClient(TestServer(server_app)) as client:
                player = await client.ws_connect("/ws")
                await player.receive_json()
                await player.receive_json()
                await player.send_json({"command": "start_demo"})
                response = await player.receive_json()
                self.assertEqual(response["type"], "error")
                self.assertFalse(app.is_demo_active)
                self.assertIsNone(app.demo_owner)
                self.assertIsNone(app.video_producer_task)

                observer = await client.ws_connect("/ws")
                await observer.receive_json()
                await observer.receive_json()
                await player.send_json({"command": "start_demo", "play_requested": True})
                response = await player.receive_json()
                self.assertEqual(response["type"], "demo_starting")
                self.assertTrue(app.is_demo_active)
                self.assertIsNotNone(app.demo_owner)

                await player.close()
                await asyncio.wait_for(producer_stopped.wait(), timeout=2)
                self.assertFalse(app.is_demo_active)
                self.assertIsNone(app.demo_owner)
                self.assertEqual(len(app.active_websockets), 1)
                await observer.close()

    async def test_motor_requires_active_game_and_connected_play_owner(self):
        owner = Mock(closed=False)
        motor = Mock()
        motor.activate_motor = AsyncMock(return_value=True)
        cases = [
            (False, None, set(), False),
            (True, None, {owner}, False),
            (False, owner, {owner}, False),
            (True, owner, set(), False),
            (True, owner, {owner}, True),
        ]
        for active, game_owner, clients, expected in cases:
            with self.subTest(active=active, owner=game_owner, clients=clients):
                motor.activate_motor.reset_mock()
                with patch.object(app, "is_demo_active", active), patch.object(
                    app, "demo_owner", game_owner
                ), patch.object(app, "active_websockets", clients), patch.object(
                    app, "MOTOR_CONTROLLER", motor
                ):
                    self.assertEqual(await app.activate_motor_for_active_game(), expected)
                self.assertEqual(motor.activate_motor.await_count, int(expected))

    async def test_closed_owner_cannot_activate_motor(self):
        owner = Mock(closed=True)
        motor = Mock(activate_motor=AsyncMock())
        with patch.object(app, "is_demo_active", True), patch.object(
            app, "demo_owner", owner
        ), patch.object(app, "active_websockets", {owner}), patch.object(
            app, "MOTOR_CONTROLLER", motor
        ):
            self.assertFalse(await app.activate_motor_for_active_game())
        motor.activate_motor.assert_not_awaited()

    async def test_stop_disarms_even_when_demo_is_inactive(self):
        with patch.object(app, "is_demo_active", False), patch.object(app, "demo_owner", Mock()):
            self.assertFalse(await app.stop_active_demo())
            self.assertIsNone(app.demo_owner)

    async def test_cancel_waits_for_blocking_worker_to_finish(self):
        started = threading.Event()
        release_worker = threading.Event()

        def blocking_work():
            started.set()
            release_worker.wait(timeout=2)
            return "finished"

        task = asyncio.create_task(app.run_blocking_to_completion(blocking_work))
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        task.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        release_worker.set()

        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_cancel_cleanup_receives_worker_result(self):
        started = threading.Event()
        release_worker = threading.Event()
        cleanup_results = []

        def blocking_work():
            started.set()
            release_worker.wait(timeout=2)
            return "opened resource"

        task = asyncio.create_task(
            app.run_blocking_to_completion(
                blocking_work,
                cancel_cleanup=cleanup_results.append,
            )
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        task.cancel()
        release_worker.set()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(cleanup_results, ["opened resource"])

    async def test_stop_cancels_active_demo_task(self):
        async def demo_task():
            await asyncio.Event().wait()

        app.is_demo_active = True
        app.video_producer_task = asyncio.create_task(demo_task())

        self.assertTrue(await app.stop_active_demo())
        self.assertFalse(app.is_demo_active)
        self.assertIsNone(app.video_producer_task)

    async def test_camera_start_failure_clears_server_active_state(self):
        with patch.object(app, "models_initialized_successfully", True), patch.object(
            app, "open_camera_device", return_value=None
        ):
            app.is_demo_active = True
            await app.video_stream_producer_loop()

        self.assertFalse(app.is_demo_active)


class SettingsValidationTests(unittest.TestCase):
    def test_valid_settings_are_whitelisted_and_normalized(self):
        validated, error = app.validate_settings_update(
            {"spin_pwm_value": 200, "happy_time_ms": 2500, "unrecognized": "ignored"}
        )

        self.assertIsNone(error)
        self.assertEqual(validated, {"spin_pwm_value": 200, "happy_time_ms": 2500.0})

    def test_invalid_consensus_settings_are_rejected(self):
        validated, error = app.validate_settings_update(
            {"emotion_consensus_window_size": 4, "emotion_consensus_required_count": 5}
        )

        self.assertIsNone(validated)
        self.assertIn("cannot exceed", error)

    def test_non_finite_numeric_setting_is_rejected(self):
        validated, error = app.validate_settings_update({"happy_time_ms": float("nan")})

        self.assertIsNone(validated)
        self.assertIn("Invalid value", error)


if __name__ == "__main__":
    unittest.main()
