"""
Robotics Perception and Control System
Advanced robotics perception and control for autonomous systems
"""

from .perception.camera_system import CameraInterface, StereoCamera, MultiCameraSystem
from .perception.object_detector import ObjectDetectionSystem, DetectedObject, TrackedObject
from .control.pid_controller import PIDController, MultiAxisPIDController

__version__ = "1.0.0"
__author__ = "Robotics Team"

__all__ = [
    'CameraInterface',
    'StereoCamera', 
    'MultiCameraSystem',
    'ObjectDetectionSystem',
    'DetectedObject',
    'TrackedObject',
    'PIDController',
    'MultiAxisPIDController'
]