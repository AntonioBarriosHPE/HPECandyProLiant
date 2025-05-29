# temptest/hardware_stubs.py
import platform
import logging
import time # We'll use this for the dummy dispense

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

class DummyGPIOControl:
    """
    A dummy class to simulate the MCP2221 GPIO control for candy dispensing.
    """
    def __init__(self):
        logging.info("DummyGPIOControl initialized: EL300 GPIO/MCP2221 not detected or not on Windows. Using dummy GPIO.")
        self._is_dispensing = False

    def _set_gpio_pins(self, pin_values_bytes):
        """
        Simulates setting GPIO pin values.
        The original C++ code writes 0x00 (all low) to turn ON (sink current)
        and 0xFF (all high) to turn OFF.
        """
        if pin_values_bytes == b'\x00':
            logging.info("DummyGPIO: Pins set to LOW (simulating dispenser ON)")
            self._is_dispensing = True
        elif pin_values_bytes == b'\xFF':
            logging.info("DummyGPIO: Pins set to HIGH (simulating dispenser OFF)")
            self._is_dispensing = False
        else:
            logging.warning(f"DummyGPIO: Received unknown pin values: {pin_values_bytes.hex()}")

    def dispense_candy(self, duration_seconds=2.0):
        """
        Simulates dispensing candy for a given duration.
        The original candy_duration in main.cpp seems to be in milliseconds.
        The JavaScript candyDurationRequirement is 3000ms or 5000ms.
        Let's stick to seconds for clarity here, convert if needed when calling.
        """
        if self._is_dispensing:
            logging.warning("DummyGPIO: Dispenser already active, ignoring new request.")
            return

        logging.info(f"DummyGPIO: --- CANDY DISPENSE START (for {duration_seconds:.1f} seconds) ---")
        self._set_gpio_pins(b'\x00') # Turn ON
        
        # Simulate the duration the candy motor would run
        # In a real scenario, this might be a blocking call or managed by a thread/timer
        # For a simple stub, a sleep is fine.
        # time.sleep(duration_seconds) # Avoid blocking the main thread if this is called directly

        # We'll assume the calling logic will handle turning it off.
        # For now, this method just logs the start. The 'off' signal will come separately.
        # Or, if we want this to be self-contained for the duration:
        # print(f"DummyGPIO: Dispensing for {duration_seconds}s...")
        # time.sleep(duration_seconds)
        # self._set_gpio_pins(b'\xFF') # Turn OFF
        # print(f"DummyGPIO: --- CANDY DISPENSE END ---")

    def turn_dispenser_on(self):
        """Simulates turning the dispenser motor ON."""
        if not self._is_dispensing:
            logging.info("DummyGPIO: --- CANDY DISPENSER MOTOR ON ---")
            self._set_gpio_pins(b'\x00')
        else:
            logging.info("DummyGPIO: Dispenser motor already ON.")

    def turn_dispenser_off(self):
        """Simulates turning the dispenser motor OFF."""
        if self._is_dispensing:
            logging.info("DummyGPIO: --- CANDY DISPENSER MOTOR OFF ---")
            self._set_gpio_pins(b'\xFF')
        else:
            logging.info("DummyGPIO: Dispenser motor already OFF.")


def is_el300_present():
    """
    Checks if the system is likely an HPE EL300.
    For macOS development, this will always return False.
    """
    if platform.system() == "Windows":
        # On a real EL300, you might check for specific drivers, WMI info, or device IDs.
        # For example, check if the MCP2221 DLL can be loaded.
        try:
            # A more robust check would involve trying to load the actual DLL
            # import ctypes
            # mcp_dll = ctypes.cdll.LoadLibrary("mcp2221_dll_um_x64.dll") # Or the path where it's expected
            # logging.info("MCP2221 DLL found. Assuming EL300-like environment for GPIO.")
            # return True # If DLL loads, assume present.
            # For now, we'll keep it simple for the stub.
            # If you were to run this script on Windows without the DLL, this would need adjustment.
            # For the macOS stub purpose, this part isn't critical.
            pass
        except OSError:
            logging.info("MCP2221 DLL not found on Windows system.")
            return False # If DLL doesn't load
        # For now, on Windows, let's assume it might be an EL300 if we add a specific check later
        # but for the primary purpose of this stub, it's for non-Windows.
        logging.warning("Running on Windows, but EL300 specific hardware check is not fully implemented in this stub.")
        return False # Default to False for Windows unless a real check is added
    logging.info(f"Platform is {platform.system()}. Not an EL300 for GPIO purposes.")
    return False

# Instantiate the dummy GPIO control globally for easy import
# If `is_el300_present()` was True and we had a `RealGPIOControl` class,
# we could conditionally instantiate it here.
GPIO_CONTROLLER = DummyGPIOControl()

if __name__ == "__main__":
    # Example usage (for testing this stub directly)
    print(f"Is EL300 present? {is_el300_present()}")

    controller = DummyGPIOControl() # or GPIO_CONTROLLER
    
    print("\nSimulating candy dispense command (ON signal):")
    controller.turn_dispenser_on()
    print(f"Is dispenser active? {controller._is_dispensing}")
    
    print("\nWaiting for 2 seconds (simulating happy duration)...")
    time.sleep(2)
    
    print("\nSimulating candy dispense OFF signal:")
    controller.turn_dispenser_off()
    print(f"Is dispenser active? {controller._is_dispensing}")

    print("\nSimulating self-contained dispense_candy (legacy if needed):")
    # This version of dispense_candy in the stub doesn't auto-off after duration to mimic C++ behavior
    # where separate on/off signals are likely given based on timer logic.
    # If the `main.cpp` logic sends an "ON" and then later an "OFF" this is more appropriate.
    controller.dispense_candy(duration_seconds=1.5) 
    # The actual dispense_candy was used as inspiration for on/off logic
    # The original C++ `main.cpp` seems to set GPIO low (0x00) to start and high (0xFF) to stop.
    # The duration logic is handled by timers in the C++ or JS.