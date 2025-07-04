"""
Perception Module
Computer vision and sensor fusion for robotics
"""

from .camera_system import CameraInterface, StereoCamera, MultiCameraSystem, CameraCalibration
from .object_detector import (
    ObjectDetectionSystem, DetectedObject, TrackedObject,
    ColorDetector, TemplateDetector, ContourDetector,
    FeatureDetector, OpticalFlowTracker, MultiObjectTracker
)

__all__ = [
    'CameraInterface',
    'StereoCamera',
    'MultiCameraSystem', 
    'CameraCalibration',
    'ObjectDetectionSystem',
    'DetectedObject',
    'TrackedObject',
    'ColorDetector',
    'TemplateDetector',
    'ContourDetector',
    'FeatureDetector',
    'OpticalFlowTracker',
    'MultiObjectTracker'
]