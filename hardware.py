# hardware.py
import asyncio
import logging
import platform
import serial # Ensure this is imported
import serial.tools.list_ports
from serial.serialutil import PortNotOpenError # For specific exception handling
import threading
import time
from abc import ABC, abstractmethod
import functools

# Setup loggers for different hardware components
logger = logging.getLogger("hpe_candy_proliant.hardware")
dummy_logger = logging.getLogger("hpe_candy_proliant.dummy_motor")
arduino_logger = logging.getLogger("hpe_candy_proliant.arduino_motor")

class MotorControllerInterface(ABC):
    """
    Interface for motor controllers (Arduino or Dummy).
    Defines the contract for how the main application interacts with motor hardware.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establishes connection to the motor controller. Returns True on success."""
        pass

    @abstractmethod
    def disconnect(self):
        """Closes the connection to the motor controller."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the controller is connected and responsive, False otherwise."""
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        """Returns True if the motor is currently in an activation cycle."""
        pass

    @abstractmethod
    async def configure_spin_pwm(self, pwm_value: int) -> bool:
        """Configures the motor's spin PWM (0-255). Returns True on success."""
        pass

    @abstractmethod
    async def configure_run_duration(self, duration_ms: int) -> bool:
        """Configures the motor's run duration in milliseconds. Returns True on success."""
        pass

    @abstractmethod
    async def activate_motor(self) -> bool:
        """
        Activates the motor in a non-blocking, fire-and-forget manner.
        Returns True immediately if the activation command was successfully sent.
        The motor cycle will run in the background.
        """
        pass

# ─────────────────────────────────────────────────────────────────────────────

class DummyMotorController(MotorControllerInterface):
    """
    A dummy motor controller that simulates Arduino behavior for testing without hardware.
    Implements the MotorControllerInterface with non-blocking activation.
    """
    def __init__(self):
        dummy_logger.info("DummyMotorController initialized.")
        self._is_busy = False
        self._spin_pwm = 100
        self._run_duration_ms = 2000
        self._connected = False

    async def connect(self) -> bool:
        dummy_logger.info("DummyMotorController: connect() called.")
        self._connected = True
        return True

    def disconnect(self):
        dummy_logger.info("DummyMotorController: disconnect() called.")
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def is_busy(self) -> bool:
        return self._is_busy

    async def configure_spin_pwm(self, pwm_value: int) -> bool:
        if not self.is_connected(): return False
        self._spin_pwm = max(0, min(255, pwm_value))
        dummy_logger.info(f"DummyMotorController: Spin PWM set to {self._spin_pwm}")
        return True

    async def configure_run_duration(self, duration_ms: int) -> bool:
        if not self.is_connected(): return False
        self._run_duration_ms = max(10, min(30000, duration_ms))
        dummy_logger.info(f"DummyMotorController: Run duration set to {self._run_duration_ms}ms")
        return True

    async def _dummy_motor_cycle(self):
        """The background task that simulates the motor running."""
        self._is_busy = True
        dummy_logger.info(f"--- DUMMY MOTOR ON (PWM: {self._spin_pwm}, Duration: {self._run_duration_ms}ms) ---")
        try:
            await asyncio.sleep(self._run_duration_ms / 1000.0)
        finally:
            self._is_busy = False
            dummy_logger.info("--- DUMMY MOTOR OFF ---")
            dummy_logger.info("Dummy: DONE motor cycle complete (simulated)")

    async def activate_motor(self) -> bool:
        if not self.is_connected():
            dummy_logger.warning("DummyMotorController: Cannot activate motor, not connected.")
            return False
        if self._is_busy:
            dummy_logger.warning("DummyMotorController: Motor already busy.")
            return False

        dummy_logger.info("DummyMotorController: Firing motor cycle as a background task.")
        asyncio.create_task(self._dummy_motor_cycle())
        return True # Return immediately

# ─────────────────────────────────────────────────────────────────────────────

class ArduinoMotorController(MotorControllerInterface):
    """
    Controls the motor by sending serial commands to an Arduino.
    Implements the MotorControllerInterface with non-blocking activation.
    """
    def __init__(self, port: str, baud_rate: int = 9600, serial_timeout: float = 0.1):
        self.port = port
        self.baud_rate = baud_rate
        self.serial_timeout_for_ops = serial_timeout
        self.serial_conn = None
        self._is_reader_active = False
        self.reader_thread = None
        self.response_queue = asyncio.Queue()
        self.command_lock = asyncio.Lock() # For configuration commands
        self._activation_lock = asyncio.Lock() # Specifically for the activate_motor sequence
        self._is_busy = False # Public flag to indicate motor cycle is active
        self._current_spin_pwm = 100
        self._current_run_duration_ms = 1000
        self.loop = None

    @staticmethod
    def get_auto_detect_com_port(device_keyword=None):
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("[ERROR] No physical COM ports found on this system.")
            return None

        valid_candidates = []

        # Step 1: Scan descriptions, hardware IDs, and manufacturer details
        for port in ports:
            # Check description, hardware ID string, and manufacturer details
            desc = (port.description or "").lower()
            hwid = (port.hwid or "").lower()
            mfg = getattr(port, 'manufacturer', '') or ""
            mfg = mfg.lower()

            # Print everything found to help you debug in your Python log files
            print(f"[DEBUG] Found Port: {port.device} | Desc: {port.description} | MFG: {port.manufacturer}")

            # Match explicitly specified keywords or standard microcontroller hardware flags
            is_match = (
                (device_keyword and device_keyword.lower() in desc) or
                (device_keyword and device_keyword.lower() in mfg) or
                "usb-serial" in desc or
                "ch340" in desc or       # Common cheap clone chips
                "cp210" in desc or       # NodeMCU / ESP8266 chips
                "ftdi" in desc           # Official FTDI chips
            )

            if is_match:
                valid_candidates.append(port.device)

        # Fallback: if no keyword matched, test all system ports instead of blinding grabbing index 0
        if not valid_candidates:
            print("[WARN] No obvious microcontroller found by keyword. Testing all available ports...")
            valid_candidates = [p.device for p in ports]

        # Step 2: Actively test connection stability on candidates (No more blind index selection!)
        for device_path in valid_candidates:
            try:
                # Try opening the port with a short timeout to see if it responds or is already blocked
                test_serial = serial.Serial(device_path, baudrate=9600, timeout=1)
                time.sleep(1)  # Allow Arduino a moment to toggle DTR/reset
                test_serial.close()

                print(f"[SUCCESS] Found working and responsive COM Port: [ {device_path} ]")
                return device_path
            except (serial.SerialException, OSError) as e:
                print(f"[INFO] Skipping {device_path}: Port is busy, locked, or unresponsive. ({str(e)})")

        print("[ERROR] Could not find any free, working COM ports to communicate with.")
        return None

    async def connect(self) -> bool:
        arduino_logger.info(f"Attempting to connect to Arduino on {self.port} at {self.baud_rate} baud.")
        if self.is_connected():
            arduino_logger.info("Already connected.")
            return True

        self.loop = asyncio.get_event_loop()

        try:
            serial_creator = functools.partial(serial.Serial, self.port, self.baud_rate, timeout=self.serial_timeout_for_ops)
            self.serial_conn = await self.loop.run_in_executor(None, serial_creator)

            def apply_dtr_fix():
                self.serial_conn.dtr = False
                time.sleep(0.4)
                self.serial_conn.reset_input_buffer()
                self.serial_conn.dtr = True
            await self.loop.run_in_executor(None, apply_dtr_fix)

            self._is_reader_active = True
            self.reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self.reader_thread.start()
            arduino_logger.info(f"Serial port {self.port} opened. Reader thread started.")

            ready_timeout_s = 4.0
            arduino_logger.info(f"Waiting for 'ARDUINO READY' for up to {ready_timeout_s}s...")
            try:
                while True: # Loop to consume startup messages
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=ready_timeout_s)
                    if "ARDUINO READY" in response:
                        arduino_logger.info("'ARDUINO READY' signal received. Connection confirmed.")
                        return True
            except asyncio.TimeoutError:
                arduino_logger.warning("'ARDUINO READY' not seen. Assuming it's already running. Proceeding.")
                return True

        except Exception as e:
            arduino_logger.error(f"ArduinoMotorController: Error during connect: {e}", exc_info=True)
            self.disconnect() # Ensure cleanup
            return False

    def disconnect(self):
        arduino_logger.info("Disconnecting from Arduino...")
        self._is_reader_active = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        self.reader_thread = None
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.serial_conn = None
        while not self.response_queue.empty(): self.response_queue.get_nowait()
        arduino_logger.info("Arduino disconnected.")

    def is_connected(self) -> bool:
        return self.serial_conn is not None and self.serial_conn.is_open and self._is_reader_active

    def is_busy(self) -> bool:
        return self._is_busy

    def _serial_reader_loop(self):
        arduino_logger.info(f"Serial reader thread started for {self.port}.")
        buffer = ''
        while self._is_reader_active:
            try:
                if not (self.serial_conn and self.serial_conn.is_open):
                    break
                raw_data = self.serial_conn.read(self.serial_conn.in_waiting or 1)
                if raw_data:
                    data_str = raw_data.decode('utf-8', errors='replace')
                    buffer += data_str
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line and self.loop and self.loop.is_running():
                            asyncio.run_coroutine_threadsafe(self.response_queue.put(line), self.loop)
            except serial.SerialException as e:
                arduino_logger.error(f"SerialException in reader loop for {self.port}: {e}. Stopping reader.")
                break
            except Exception as e:
                arduino_logger.error(f"Unexpected exception in reader loop: {e}", exc_info=True)
                break
            time.sleep(0.005)
        self._is_reader_active = False
        arduino_logger.info(f"Serial reader thread for {self.port} finished.")

    async def _send_command_and_wait_for_response(self, command: str, expected_prefix: str, timeout_s: float = 3.0) -> tuple[bool, str | None]:
        async with self.command_lock:
            if not self.is_connected(): return False, None
            while not self.response_queue.empty(): self.response_queue.get_nowait()
            await self.loop.run_in_executor(None, self.serial_conn.write, command.encode('utf-8'))
            try:
                while True:
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=timeout_s)
                    if response.startswith(expected_prefix): return True, response
            except asyncio.TimeoutError:
                arduino_logger.warning(f"Timeout waiting for '{expected_prefix}' after sending '{command.strip()}'.")
                return False, None

    async def configure_spin_pwm(self, pwm_value: int) -> bool:
        self._current_spin_pwm = max(0, min(255, pwm_value))
        cmd = f"SPIN:{self._current_spin_pwm}\n"
        success, _ = await self._send_command_and_wait_for_response(cmd, f"ACK SPIN set to {self._current_spin_pwm}")
        if success: arduino_logger.info(f"Arduino: SPIN successfully configured to {self._current_spin_pwm}")
        return success

    async def configure_run_duration(self, duration_ms: int) -> bool:
        self._current_run_duration_ms = max(10, min(30000, duration_ms))
        cmd = f"TIME:{self._current_run_duration_ms}\n"
        success, _ = await self._send_command_and_wait_for_response(cmd, f"ACK TIME set to {self._current_run_duration_ms}")
        if success: arduino_logger.info(f"Arduino: TIME successfully configured to {self._current_run_duration_ms} ms")
        return success

    async def _handle_motor_cycle_feedback(self):
        """
        Internal background task to listen for Arduino feedback during a motor cycle
        and manage the _is_busy flag.
        """
        try:
            # Stage 1: Wait for "CMD SMILE received"
            cmd_ack_received = False
            cmd_ack_timeout_s = 3.0
            try:
                while True: # Loop to find the right message
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=cmd_ack_timeout_s)
                    if response.startswith("CMD SMILE received"):
                        arduino_logger.info("BG Task: 'CMD SMILE received' acknowledgment OK.")
                        cmd_ack_received = True
                        break
            except asyncio.TimeoutError:
                arduino_logger.error(f"BG Task: Timeout ({cmd_ack_timeout_s}s) waiting for 'CMD SMILE received'.")
                return # Exits the background task

            if not cmd_ack_received: return # Should be unreachable, but safe

            # Stage 2: Wait for "DONE motor cycle complete"
            done_received = False
            done_timeout_s = (self._current_run_duration_ms / 1000.0) + 5.0 # Motor duration + 5s buffer
            try:
                while True:
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=done_timeout_s)
                    if response.startswith("DONE motor cycle complete"):
                        arduino_logger.info("BG Task: Motor cycle complete ('DONE' received).")
                        done_received = True
                        break
            except asyncio.TimeoutError:
                arduino_logger.warning(f"BG Task: Timeout ({done_timeout_s:.1f}s) waiting for 'DONE'. Confirmation missing.")

        finally:
            # This is crucial: no matter what happens, un-set the busy flag
            self._is_busy = False
            arduino_logger.info("BG Task: Cycle finished. Motor is no longer busy.")

    async def activate_motor(self) -> bool:
        """
        Non-blocking motor activation. Sends command, starts a background feedback listener,
        and returns immediately.
        """
        async with self._activation_lock: # Ensure this check-and-set is atomic
            if not self.is_connected():
                arduino_logger.error("Arduino: Cannot activate motor, not connected.")
                return False
            if self._is_busy:
                arduino_logger.warning("Arduino: Cannot activate motor, already busy.")
                return False

            # Set busy flag immediately
            self._is_busy = True

        # If we get here, we are clear to send the command
        try:
            # Clear any old messages from queue before sending
            while not self.response_queue.empty(): self.response_queue.get_nowait()

            # Send the command
            cmd = "SMILE\n"
            await self.loop.run_in_executor(None, self.serial_conn.write, cmd.encode('utf-8'))
            arduino_logger.info(f"Sent '{cmd.strip()}' to Arduino. Firing background listener.")

            # Start the background task to handle feedback and reset the busy flag
            asyncio.create_task(self._handle_motor_cycle_feedback())

            # Return True immediately to unblock the main application
            return True

        except Exception as e:
            arduino_logger.error(f"Arduino: Failed to send SMILE command: {e}")
            # If sending failed, we must reset the busy flag we just set
            self._is_busy = False
            return False