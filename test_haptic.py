from dataclasses import dataclass, field
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from pyOpenHaptics.hd_device import HapticDevice
import time

@dataclass
class DeviceState:
    button: bool = False
    position: list = field(default_factory=list)
    force: list = field(default_factory=list)

device_state = DeviceState()

@hd_callback
def device_callback():
    global device_state
    """
    Callback function for the haptic device.
    """
    # Get the current position of the device
    transform = hd.get_transform()
    device_state.position = [transform[3][0], -transform[3][1], transform[3][2]]
    # Set the force to the device
    hd.set_force(device_state.force)
    # Get the current state of the device buttons
    button = hd.get_buttons()
    device_state.button = True if button == 1 else False

def main():
    device = HapticDevice(device_name="Default Device", callback=device_callback)
    for i in range(1000):
        print(device_state.button)
        time.sleep(0.1)
        
    device.close()
if __name__ == "__main__":
    main()
