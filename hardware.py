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
        Activates the motor for one cycle using the currently stored configuration.
        This is a blocking async call that should only return after the motor cycle is complete.
        Returns True if the full cycle completed successfully.
        """
        pass

# ─────────────────────────────────────────────────────────────────────────────

class DummyMotorController(MotorControllerInterface):
    """
    A dummy motor controller that simulates Arduino behavior for testing without hardware.
    Implements the MotorControllerInterface.
    """
    def __init__(self):
        dummy_logger.info("DummyMotorController initialized.")
        self._is_motor_active = False
        self._spin_pwm = 100
        self._run_duration_ms = 2000
        self._connected = False

    async def connect(self) -> bool:
        dummy_logger.info("DummyMotorController: connect() called.")
        self._connected = True
        dummy_logger.info("DummyMotorController: Simulated 'ARDUINO READY'")
        dummy_logger.info(f"DummyMotorController: Defaults SPIN={self._spin_pwm} TIME={self._run_duration_ms} ms")
        return True

    def disconnect(self):
        dummy_logger.info("DummyMotorController: disconnect() called.")
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    async def configure_spin_pwm(self, pwm_value: int) -> bool:
        if not self.is_connected():
            dummy_logger.warning("DummyMotorController: Cannot configure spin, not connected.")
            return False
        self._spin_pwm = max(0, min(255, pwm_value))
        dummy_logger.info(f"DummyMotorController: Spin PWM set to {self._spin_pwm}")
        dummy_logger.info(f"DummyMotorController: Simulated 'ACK SPIN set to {self._spin_pwm}'")
        return True

    async def configure_run_duration(self, duration_ms: int) -> bool:
        if not self.is_connected():
            dummy_logger.warning("DummyMotorController: Cannot configure duration, not connected.")
            return False
        self._run_duration_ms = max(10, min(30000, duration_ms))
        dummy_logger.info(f"DummyMotorController: Run duration set to {self._run_duration_ms}ms")
        dummy_logger.info(f"DummyMotorController: Simulated 'ACK TIME set to {self._run_duration_ms} ms'")
        return True

    async def activate_motor(self) -> bool:
        if not self.is_connected():
            dummy_logger.warning("DummyMotorController: Cannot activate motor, not connected.")
            return False
        if self._is_motor_active:
            dummy_logger.warning("DummyMotorController: Motor already active (simulated).")
            return False

        dummy_logger.info("DummyMotorController: CMD SMILE received – activating motor (simulated)")
        dummy_logger.info(f"DummyMotorController: --- MOTOR ON (PWM: {self._spin_pwm}, Duration: {self._run_duration_ms}ms) ---")
        self._is_motor_active = True
        
        await asyncio.sleep(self._run_duration_ms / 1000.0)
        
        self._is_motor_active = False
        dummy_logger.info("DummyMotorController: --- MOTOR OFF (Simulated) ---")
        dummy_logger.info("DummyMotorController: DONE motor cycle complete (simulated)")
        return True

# ─────────────────────────────────────────────────────────────────────────────

class ArduinoMotorController(MotorControllerInterface):
    """
    Controls the motor by sending serial commands to an Arduino.
    Implements the MotorControllerInterface.
    """
    def __init__(self, port: str, baud_rate: int = 9600, serial_timeout: float = 0.1): # serial_timeout for read operations
        self.port = port
        self.baud_rate = baud_rate
        self.serial_timeout_for_ops = serial_timeout # Timeout for individual read/write ops inside serial.Serial
        self.serial_conn = None
        self._is_reader_active = False
        self.reader_thread = None
        self.response_queue = asyncio.Queue()
        self.command_lock = asyncio.Lock() # Ensures one full command sequence (send+ack) at a time
        self._current_spin_pwm = 100  # Default value, will be synced with Arduino
        self._current_run_duration_ms = 1000 # Default value, will be synced
        self.loop = None # To store event loop for threadsafe calls

    @staticmethod
    def get_auto_detect_com_port(device_keyword=None):
        ports = serial.tools.list_ports.comports()
        if not ports:
            print(f"[INFO] No COM ports found")
            return None

        if device_keyword:
            for port in ports:
                if device_keyword.lower() in port.description.lower():
                    print(f"[INFO] found Matching Device: {port.description} on {port.device}")
                    return port.device

        print(f"[INFO] Assigning First Available Port: {ports[0].description} on=> [ {ports[0].device} ]")
        return ports[0].device

    async def connect(self) -> bool:
        arduino_logger.info(f"Attempting to connect to Arduino on {self.port} at {self.baud_rate} baud.")
        if self.is_connected():
            arduino_logger.info("Already connected.")
            return True
        
        self.loop = asyncio.get_event_loop()

        try:
            serial_creator = functools.partial(
                serial.Serial,
                self.port,
                self.baud_rate,
                timeout=self.serial_timeout_for_ops
            )
            self.serial_conn = await self.loop.run_in_executor(
                None,             
                serial_creator    
            )
            
            # DTR fix for CH340 chips - wrapped in executor to avoid blocking
            def apply_dtr_fix():
                self.serial_conn.dtr = False   # drop DTR
                time.sleep(0.4)               # let the MCU finish reset
                self.serial_conn.reset_input_buffer()
                self.serial_conn.dtr = True    # raise DTR, start normal comms
            
            await self.loop.run_in_executor(None, apply_dtr_fix)

            self._is_reader_active = True
            self.reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self.reader_thread.start()
            arduino_logger.info(f"Serial port {self.port} opened. Reader thread started.")

            # --- MODIFIED "ARDUINO READY" LOGIC ---
            # Try to get "ARDUINO READY" for a short period, but don't fail catastrophically if not seen.
            # Assume it might be already running.
            short_ready_timeout_s = 4.0  # Increased from 2.0 as suggested in your original document
            connect_proceed_anyway_after_s = 0.5 # Time to wait for *any* serial activity before just proceeding
            
            start_time = time.monotonic()
            arduino_ready_explicitly_received = False
            initial_lines_count = 0

            arduino_logger.info(f"Attempting to detect 'ARDUINO READY' for ~{short_ready_timeout_s}s...")
            
            # First, quickly check if any data comes through. If the port just opened and Arduino resets,
            # messages should arrive.
            time_waited_for_any_data = 0
            while time_waited_for_any_data < connect_proceed_anyway_after_s:
                if not self.response_queue.empty():
                    break # Data is available, proceed to process it
                await asyncio.sleep(0.05)
                time_waited_for_any_data += 0.05
            
            # Now, process available data for short_ready_timeout_s
            while (time.monotonic() - start_time) < short_ready_timeout_s:
                try:
                    # Poll the queue without blocking the connect method for too long per attempt
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=0.1) 
                    arduino_logger.info(f"Arduino Connect (Initial Recv): {response}")
                    initial_lines_count += 1
                    if "ARDUINO READY" in response:
                        arduino_ready_explicitly_received = True
                        arduino_logger.info(f"'ARDUINO READY' signal received from {self.port}.")
                        # Optionally, you could break here or continue to consume other startup lines
                        # For now, let's just note it and let the loop timeout or consume more.
                except asyncio.TimeoutError:
                    # No message in queue this iteration, see if overall timeout reached
                    if (time.monotonic() - start_time) >= short_ready_timeout_s:
                        break # Exit outer while loop
                    # else continue, more time left
                except Exception as e:
                    arduino_logger.error(f"Exception during initial 'ARDUINO READY' check: {e}")
                    break # Exit on other exceptions

            if arduino_ready_explicitly_received:
                arduino_logger.info(f"Arduino connection established and 'ARDUINO READY' confirmed. ({initial_lines_count} initial lines read).")
            else:
                arduino_logger.warning(f"'ARDUINO READY' signal not explicitly detected within ~{short_ready_timeout_s}s ({initial_lines_count} initial lines read). Assuming Arduino may already be running. Proceeding with connection.")
            
            # Regardless of "ARDUINO READY", if port opened and reader started, consider it "connected"
            # The success of subsequent commands will be the real test.
            return True 
            # --- END OF MODIFIED "ARDUINO READY" LOGIC ---

        except serial.SerialException as e:
            arduino_logger.error(f"ArduinoMotorController: Serial connection error on {self.port}: {e}")
            self.serial_conn = None 
            return False
        except Exception as e: 
            arduino_logger.error(f"ArduinoMotorController: Unspecified error during connect: {e}", exc_info=True)
            if self.serial_conn and self.serial_conn.is_open:
                try: 
                    self.serial_conn.close()
                except: 
                    pass 
            self.serial_conn = None
            return False

    def disconnect(self):
        arduino_logger.info("Disconnecting from Arduino...")
        self._is_reader_active = False # Signal reader thread to stop
        if self.reader_thread and self.reader_thread.is_alive():
            try:
                self.reader_thread.join(timeout=1.0) # Wait for thread to finish
                if self.reader_thread.is_alive():
                    arduino_logger.warning("Reader thread did not terminate in time.")
            except Exception as e:
                 arduino_logger.error(f"Error joining reader thread: {e}")
        self.reader_thread = None

        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                arduino_logger.info(f"Serial connection to {self.port} closed.")
            except Exception as e:
                arduino_logger.error(f"Error closing serial port {self.port}: {e}")
        self.serial_conn = None
        # Clear the queue in case of disconnect/reconnect
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        arduino_logger.info("Arduino disconnected.")


    def is_connected(self) -> bool:
        # Check reader thread status as well, as port might be open but reader failed
        return self.serial_conn is not None and self.serial_conn.is_open and self._is_reader_active and self.reader_thread.is_alive()


    def _serial_reader_loop(self):
        """Dedicated thread for reading from serial and putting lines onto an asyncio.Queue."""
        buffer = ''
        arduino_logger.info(f"Serial reader thread started for {self.port}.")
        try: # Add a try/except block for the whole loop for robustness
            while self._is_reader_active:
                if not (self.serial_conn and self.serial_conn.is_open):
                    arduino_logger.warning("Serial connection not available in reader loop. Stopping reader.")
                    self._is_reader_active = False
                    break

                try:
                    # Try a direct blocking read with a timeout.
                    # This is simpler than relying on in_waiting for initial debugging.
                    # The timeout on serial.Serial object (self.serial_timeout_for_ops) should apply here.
                    # If self.serial_timeout_for_ops is 0.1s, this read will block for max 0.1s if no data.
                    
                    # DEBUG: How many bytes are available according to in_waiting?
                    # available_bytes = self.serial_conn.in_waiting
                    # if available_bytes > 0:
                    #    arduino_logger.debug(f"Reader: {available_bytes} bytes in_waiting.")

                    # Attempt to read at least 1 byte, will block up to serial_timeout_for_ops
                    # If it returns empty bytes b'', it means it timed out.
                    raw_data = self.serial_conn.read(1) # Read 1 byte
                    if self.serial_conn.in_waiting > 0: # If more bytes are available after reading 1
                        raw_data += self.serial_conn.read(self.serial_conn.in_waiting) # Read the rest

                    if raw_data: # If any data was read
                        arduino_logger.debug(f"Reader RAW BYTES: {raw_data!r}") # Log raw bytes
                        try:
                            data_str = raw_data.decode('utf-8', errors='replace') # Use 'replace' for debugging
                            arduino_logger.debug(f"Reader DECODED: {data_str!r}")
                            buffer += data_str
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                arduino_logger.debug(f"Reader PARSED LINE: {line!r}")
                                if line:
                                    if self.loop and self.loop.is_running():
                                        asyncio.run_coroutine_threadsafe(self.response_queue.put(line), self.loop)
                                    else:
                                        arduino_logger.warning("Event loop not running, cannot queue serial message. Stopping reader.")
                                        self._is_reader_active = False
                                        break 
                        except UnicodeDecodeError as ude:
                            arduino_logger.error(f"Reader UnicodeDecodeError: {ude}. Raw: {raw_data!r}")
                    # else:
                        # No data read in this attempt (timed out if read(1) was blocking)
                        # arduino_logger.debug("Reader: No data from read(1) this cycle.")
                        # The main loop has a time.sleep if we don't read, so this is fine.
                        # However, a blocking read with timeout is better.

                except serial.SerialTimeoutException: # This can happen if timeout is set and read times out
                    arduino_logger.debug("Reader: serial.SerialTimeoutException (normal if no data).")
                    # This is okay, means no data arrived within the read timeout
                    pass # Continue the loop
                except serial.SerialException as e:
                    arduino_logger.error(f"SerialException in reader loop for {self.port}: {e}. Stopping reader.")
                    if self.loop and self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.response_queue.put("SERIAL_ERROR"), self.loop)
                    self._is_reader_active = False 
                    break 
                except Exception as e:
                    arduino_logger.error(f"Unexpected exception in reader inner loop for {self.port}: {e}", exc_info=True)
                    self._is_reader_active = False # Stop on unexpected errors too
                    break
                
                if not self._is_reader_active: # Check flag again if broken from inner try
                    break
                
                time.sleep(0.005) # Very short sleep to yield, but rely on read timeout mostly

        except Exception as e:
            arduino_logger.error(f"Critical error in _serial_reader_loop: {e}", exc_info=True)
        finally:
            arduino_logger.info(f"Serial reader thread for {self.port} finished.")


    async def _send_command_and_wait_for_response(self, command: str, expected_prefix: str, response_timeout_s: float = 3.0) -> tuple[bool, str | None]:
        if not self.is_connected():
            arduino_logger.error(f"Not connected. Cannot send command: {command.strip()}")
            return False, None
        
        async with self.command_lock:
            try:
                # Clear any stale responses from queue that might have accumulated before this command
                # This is a bit aggressive, but ensures we're waiting for THIS command's response
                while not self.response_queue.empty(): self.response_queue.get_nowait()

                await self.loop.run_in_executor(None, self.serial_conn.write, command.encode('utf-8'))
                arduino_logger.debug(f"Sent to Arduino: {command.strip()}")

                start_time = time.monotonic()
                while time.monotonic() - start_time < response_timeout_s:
                    try:
                        response = await asyncio.wait_for(self.response_queue.get(), timeout=0.2) # Short poll on queue
                        arduino_logger.debug(f"Recv from Arduino: {response}")
                        
                        if response == "SERIAL_ERROR": # Special signal from reader thread
                            arduino_logger.error("Serial error detected by reader thread during command.")
                            # disconnect() should be called by the part of the code that detects reader is no longer active
                            # For now, we can signal failure.
                            return False, "SERIAL_ERROR"

                        if expected_prefix is None or response.startswith(expected_prefix): # None means any response is OK (for initial connect)
                            return True, response
                        else:
                            arduino_logger.warning(f"Received unexpected response '{response}' while waiting for '{expected_prefix}'. Still waiting.")
                    except asyncio.TimeoutError:
                        # Timeout for self.response_queue.get(), loop continues until response_timeout_s
                        pass 
                    except asyncio.QueueEmpty: # Should not happen with wait_for, but good to be aware
                        pass
                    except Exception as e:
                        arduino_logger.error(f"Exception while processing response queue: {e}")
                        return False, None 
                
                arduino_logger.warning(f"Timeout ({response_timeout_s}s) waiting for response starting with '{expected_prefix}' after sending '{command.strip()}'.")
                return False, None
            except serial.SerialException as se:
                arduino_logger.error(f"SerialException sending command '{command.strip()}': {se}")
                # Consider calling disconnect here or letting higher level logic handle it
                return False, None
            except Exception as e:
                arduino_logger.error(f"Error sending command '{command.strip()}': {e}")
                return False, None

    async def configure_spin_pwm(self, pwm_value: int) -> bool:
        self._current_spin_pwm = max(0, min(255, pwm_value))
        cmd = f"SPIN:{self._current_spin_pwm}\n"
        expected_response = f"ACK SPIN set to {self._current_spin_pwm}"
        success, _ = await self._send_command_and_wait_for_response(cmd, expected_response)
        if success:
            arduino_logger.info(f"Arduino: SPIN successfully configured to {self._current_spin_pwm}")
        else:
            arduino_logger.error(f"Arduino: Failed to configure SPIN to {self._current_spin_pwm}")
        return success

    async def configure_run_duration(self, duration_ms: int) -> bool:
        self._current_run_duration_ms = max(10, min(30000, duration_ms)) # Match Arduino constraints
        cmd = f"TIME:{self._current_run_duration_ms}\n"
        expected_response = f"ACK TIME set to {self._current_run_duration_ms}"
        success, _ = await self._send_command_and_wait_for_response(cmd, expected_response)
        if success:
            arduino_logger.info(f"Arduino: TIME successfully configured to {self._current_run_duration_ms} ms")
        else:
            arduino_logger.error(f"Arduino: Failed to configure TIME to {self._current_run_duration_ms} ms")
        return success

    async def activate_motor(self) -> bool:
        if not self.is_connected():
            arduino_logger.error("Arduino: Cannot activate motor, not connected.")
            return False
        
        cmd = "SMILE\n"
        arduino_logger.info("Arduino: Activating motor (sending SMILE)...")
        
        async with self.command_lock: # Ensure atomicity of SMILE -> CMD_ACK -> DONE sequence
            # Clear queue before sending SMILE
            while not self.response_queue.empty(): self.response_queue.get_nowait()

            # Stage 1: Send SMILE command
            try:
                await self.loop.run_in_executor(None, self.serial_conn.write, cmd.encode('utf-8'))
                arduino_logger.debug(f"Sent to Arduino: {cmd.strip()}")
            except Exception as e:
                arduino_logger.error(f"Arduino: Failed to send SMILE command: {e}")
                # self.disconnect() # Potentially disconnect if send fails catastrophically
                return False

            # Stage 2: Wait for "CMD SMILE received"
            cmd_ack_received = False
            cmd_ack_timeout_s = 3.0 
            start_time_cmd = time.monotonic()
            arduino_logger.info("Arduino: Waiting for 'CMD SMILE received' acknowledgment...")
            while time.monotonic() - start_time_cmd < cmd_ack_timeout_s:
                try:
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=0.2)
                    arduino_logger.debug(f"Arduino Recv (awaiting CMD_ACK): {response}")
                    if response == "SERIAL_ERROR":
                        arduino_logger.error("Serial error while waiting for CMD_ACK.")
                        return False # Reader thread signaled error, connection is likely dead
                    if response.startswith("CMD SMILE received"):
                        arduino_logger.info("Arduino: 'CMD SMILE received' acknowledgment OK.")
                        cmd_ack_received = True
                        break
                except asyncio.TimeoutError:
                    continue # Continue polling queue until cmd_ack_timeout_s
            
            if not cmd_ack_received:
                arduino_logger.error(f"Arduino: Timeout ({cmd_ack_timeout_s}s) waiting for 'CMD SMILE received' acknowledgment.")
                return False

            # Stage 3: Wait for "DONE motor cycle complete"
            done_received = False
            # Timeout for DONE should be slightly more than the motor run duration + buffer for serial comms
            done_timeout_s = (self._current_run_duration_ms / 1000.0) + 5.0 # Add 5s buffer
            start_time_done = time.monotonic()
            arduino_logger.info(f"Arduino: Waiting for 'DONE motor cycle complete' (timeout: {done_timeout_s:.1f}s). Motor should run for {self._current_run_duration_ms / 1000.0:.1f}s.")
            
            while time.monotonic() - start_time_done < done_timeout_s:
                try:
                    response = await asyncio.wait_for(self.response_queue.get(), timeout=0.2)
                    arduino_logger.debug(f"Arduino Recv (awaiting DONE): {response}")
                    if response == "SERIAL_ERROR":
                        arduino_logger.error("Serial error while waiting for DONE.")
                        return False
                    if response.startswith("DONE motor cycle complete"):
                        arduino_logger.info("Arduino: Motor cycle complete ('DONE' received).")
                        done_received = True
                        break
                except asyncio.TimeoutError:
                    continue # Continue polling queue until done_timeout_s
            
            if not done_received:
                arduino_logger.warning(f"Arduino: Timeout ({done_timeout_s}s) waiting for 'DONE motor cycle complete'. Motor might have run, but confirmation missing.")
                # Depending on strictness, you might return False or True with a warning.
                # For now, if DONE is not received, consider it a failure of the full cycle.
                return False
            
            return True # Both CMD_ACK and DONE were received