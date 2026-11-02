"""
Python module for simulating GelSight sensors inside Isaac Sim/Lab
"""

from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

# Isaac Lab's headless 5.1 experience does not load omni.ui. The optical
# simulator is usable without registering the optional example window.
import importlib.util

__all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData"]
if importlib.util.find_spec("omni.ui") is not None:
    from .ui_extension_example import UsdrtExamplePythonExtension
    __all__.append("UsdrtExamplePythonExtension")
