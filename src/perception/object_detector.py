"""
Object Detection Module
Advanced object detection and tracking for robotics applications
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from dataclasses import dataclass
from enum import Enum
import logging


@dataclass
class DetectedObject:
    """Detected object information"""
    id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]
    area: float
    timestamp: float
    features: Optional[np.ndarray] = None
    depth: Optional[float] = None


class TrackingState(Enum):
    """Object tracking states"""
    ACTIVE = "active"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class TrackedObject:
    """Tracked object with history"""
    id: int
    class_name: str
    current_bbox: Tuple[int, int, int, int]
    position_history: List[Tuple[float, float]]
    confidence_history: List[float]
    velocity: Tuple[float, float]
    state: TrackingState
    frames_since_update: int
    total_frames: int
    creation_time: float


class FeatureDetector:
    """Feature detection and description"""
    
    def __init__(self, detector_type: str = 'ORB'):
        """
        Initialize feature detector
        
        Args:
            detector_type: Type of detector ('ORB', 'SIFT', 'SURF', 'AKAZE')
        """
        self.detector_type = detector_type
        self.detector = self._create_detector(detector_type)
    
    def _create_detector(self, detector_type: str):
        """Create feature detector"""
        if detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=1000)
        elif detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.75) -> List:
        """
        Match features between two images
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            ratio_threshold: Ratio test threshold
            
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Create matcher
        if self.detector_type == 'ORB' or self.detector_type == 'AKAZE':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Match descriptors
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches


class ColorDetector:
    """Color-based object detection"""
    
    def __init__(self):
        """Initialize color detector"""
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'green': [(50, 50, 50), (70, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)],
            'purple': [(130, 50, 50), (170, 255, 255)]
        }
    
    def detect_color_objects(self, image: np.ndarray, color: str,
                           min_area: int = 500) -> List[DetectedObject]:
        """
        Detect objects by color
        
        Args:
            image: Input image
            color: Color name to detect
            min_area: Minimum object area
            
        Returns:
            List of detected objects
        """
        if color not in self.color_ranges:
            raise ValueError(f"Color '{color}' not supported")
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        color_range = self.color_ranges[color]
        
        if len(color_range) == 4:  # Red color (wraps around)
            mask1 = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))
            mask2 = cv2.inRange(hsv, np.array(color_range[2]), np.array(color_range[3]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w/2, y + h/2)
                
                obj = DetectedObject(
                    id=i,
                    class_name=f"{color}_object",
                    confidence=1.0,
                    bbox=(x, y, w, h),
                    center=center,
                    area=area,
                    timestamp=time.time()
                )
                detected_objects.append(obj)
        
        return detected_objects


class TemplateDetector:
    """Template matching based detection"""
    
    def __init__(self):
        """Initialize template detector"""
        self.templates = {}
    
    def add_template(self, name: str, template: np.ndarray):
        """Add template for matching"""
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.templates[name] = template
    
    def detect_template(self, image: np.ndarray, template_name: str,
                       threshold: float = 0.8) -> List[DetectedObject]:
        """
        Detect template in image
        
        Args:
            image: Input image
            template_name: Name of template to search for
            threshold: Detection threshold
            
        Returns:
            List of detected objects
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Convert image to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations above threshold
        locations = np.where(result >= threshold)
        
        detected_objects = []
        h, w = template.shape
        
        for i, (y, x) in enumerate(zip(locations[0], locations[1])):
            confidence = result[y, x]
            center = (x + w/2, y + h/2)
            
            obj = DetectedObject(
                id=i,
                class_name=template_name,
                confidence=confidence,
                bbox=(x, y, w, h),
                center=center,
                area=w * h,
                timestamp=time.time()
            )
            detected_objects.append(obj)
        
        return detected_objects


class ContourDetector:
    """Contour-based object detection"""
    
    def __init__(self):
        """Initialize contour detector"""
        pass
    
    def detect_objects(self, image: np.ndarray, min_area: int = 500,
                      max_area: int = 50000) -> List[DetectedObject]:
        """
        Detect objects using contour analysis
        
        Args:
            image: Input image
            min_area: Minimum object area
            max_area: Maximum object area
            
        Returns:
            List of detected objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w/2, y + h/2)
                
                # Calculate shape features
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                obj = DetectedObject(
                    id=i,
                    class_name="contour_object",
                    confidence=circularity,  # Use circularity as confidence
                    bbox=(x, y, w, h),
                    center=center,
                    area=area,
                    timestamp=time.time()
                )
                detected_objects.append(obj)
        
        return detected_objects


class OpticalFlowTracker:
    """Optical flow based object tracking"""
    
    def __init__(self):
        """Initialize optical flow tracker"""
        self.tracks = {}
        self.next_id = 0
        self.max_tracks = 100
        self.track_len = 10
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def track_features(self, prev_gray: np.ndarray, 
                      curr_gray: np.ndarray) -> Tuple[List, List]:
        """
        Track features between consecutive frames
        
        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            
        Returns:
            Tuple of (good_new_points, good_old_points)
        """
        # Detect corners in previous frame
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        
        if p0 is None:
            return [], []
        
        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **self.lk_params
        )
        
        # Select good points
        good_new = p1[status == 1]
        good_old = p0[status == 1]
        
        return good_new.tolist(), good_old.tolist()


class MultiObjectTracker:
    """Multi-object tracking system"""
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 50):
        """
        Initialize multi-object tracker
        
        Args:
            max_disappeared: Maximum frames object can be lost
            max_distance: Maximum distance for object association
        """
        self.next_object_id = 0
        self.tracked_objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register_object(self, center: Tuple[float, float], 
                       detected_obj: DetectedObject) -> int:
        """Register new object"""
        object_id = self.next_object_id
        
        tracked_obj = TrackedObject(
            id=object_id,
            class_name=detected_obj.class_name,
            current_bbox=detected_obj.bbox,
            position_history=[center],
            confidence_history=[detected_obj.confidence],
            velocity=(0.0, 0.0),
            state=TrackingState.ACTIVE,
            frames_since_update=0,
            total_frames=1,
            creation_time=time.time()
        )
        
        self.tracked_objects[object_id] = tracked_obj
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        
        return object_id
    
    def deregister_object(self, object_id: int):
        """Remove object from tracking"""
        if object_id in self.tracked_objects:
            self.tracked_objects[object_id].state = TrackingState.DELETED
            del self.tracked_objects[object_id]
            del self.disappeared[object_id]
    
    def update(self, detected_objects: List[DetectedObject]) -> Dict[int, TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detected_objects: List of detected objects
            
        Returns:
            Dictionary of tracked objects
        """
        # If no detections, mark all as disappeared
        if len(detected_objects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                self.tracked_objects[object_id].frames_since_update += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister_object(object_id)
                else:
                    self.tracked_objects[object_id].state = TrackingState.LOST
            
            return self.tracked_objects
        
        # Initialize object centers
        detection_centers = [obj.center for obj in detected_objects]
        
        # If no existing objects, register all detections
        if len(self.tracked_objects) == 0:
            for detection in detected_objects:
                self.register_object(detection.center, detection)
        else:
            # Compute distance matrix
            object_centers = [obj.position_history[-1] 
                            for obj in self.tracked_objects.values()]
            
            D = self._compute_distance_matrix(object_centers, detection_centers)
            
            # Assignment using Hungarian algorithm (simplified)
            assignments = self._assign_detections(D)
            
            # Update existing objects
            for (object_idx, detection_idx) in assignments:
                object_id = list(self.tracked_objects.keys())[object_idx]
                detection = detected_objects[detection_idx]
                
                # Update tracked object
                tracked_obj = self.tracked_objects[object_id]
                tracked_obj.current_bbox = detection.bbox
                tracked_obj.position_history.append(detection.center)
                tracked_obj.confidence_history.append(detection.confidence)
                tracked_obj.frames_since_update = 0
                tracked_obj.total_frames += 1
                tracked_obj.state = TrackingState.ACTIVE
                
                # Update velocity
                if len(tracked_obj.position_history) >= 2:
                    prev_pos = tracked_obj.position_history[-2]
                    curr_pos = tracked_obj.position_history[-1]
                    tracked_obj.velocity = (
                        curr_pos[0] - prev_pos[0],
                        curr_pos[1] - prev_pos[1]
                    )
                
                # Limit history length
                if len(tracked_obj.position_history) > 50:
                    tracked_obj.position_history.pop(0)
                    tracked_obj.confidence_history.pop(0)
                
                self.disappeared[object_id] = 0
            
            # Handle unassigned detections (new objects)
            unassigned_detections = set(range(len(detected_objects))) - set([d for (o, d) in assignments])
            for detection_idx in unassigned_detections:
                self.register_object(detection_centers[detection_idx], 
                                   detected_objects[detection_idx])
            
            # Handle unassigned objects (disappeared)
            unassigned_objects = set(range(len(self.tracked_objects))) - set([o for (o, d) in assignments])
            for object_idx in unassigned_objects:
                object_id = list(self.tracked_objects.keys())[object_idx]
                self.disappeared[object_id] += 1
                self.tracked_objects[object_id].frames_since_update += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister_object(object_id)
                else:
                    self.tracked_objects[object_id].state = TrackingState.LOST
        
        return self.tracked_objects
    
    def _compute_distance_matrix(self, object_centers: List, 
                                detection_centers: List) -> np.ndarray:
        """Compute distance matrix between objects and detections"""
        D = np.zeros((len(object_centers), len(detection_centers)))
        
        for i, obj_center in enumerate(object_centers):
            for j, det_center in enumerate(detection_centers):
                D[i][j] = np.sqrt((obj_center[0] - det_center[0])**2 + 
                                (obj_center[1] - det_center[1])**2)
        
        return D
    
    def _assign_detections(self, distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Simple assignment algorithm (greedy)"""
        assignments = []
        used_objects = set()
        used_detections = set()
        
        # Find minimum distances below threshold
        rows, cols = distance_matrix.shape
        for _ in range(min(rows, cols)):
            min_dist = float('inf')
            min_row, min_col = -1, -1
            
            for i in range(rows):
                if i in used_objects:
                    continue
                for j in range(cols):
                    if j in used_detections:
                        continue
                    if distance_matrix[i, j] < min_dist and distance_matrix[i, j] < self.max_distance:
                        min_dist = distance_matrix[i, j]
                        min_row, min_col = i, j
            
            if min_row != -1 and min_col != -1:
                assignments.append((min_row, min_col))
                used_objects.add(min_row)
                used_detections.add(min_col)
            else:
                break
        
        return assignments


class ObjectDetectionSystem:
    """Complete object detection and tracking system"""
    
    def __init__(self):
        """Initialize detection system"""
        self.color_detector = ColorDetector()
        self.template_detector = TemplateDetector()
        self.contour_detector = ContourDetector()
        self.feature_detector = FeatureDetector()
        self.optical_flow_tracker = OpticalFlowTracker()
        self.multi_tracker = MultiObjectTracker()
        
        self.detection_methods = {}
        self.tracking_enabled = True
        
    def register_detection_method(self, name: str, method: str, **kwargs):
        """Register detection method"""
        self.detection_methods[name] = {
            'method': method,
            'params': kwargs
        }
    
    def detect_objects(self, image: np.ndarray, 
                      methods: Optional[List[str]] = None) -> List[DetectedObject]:
        """
        Detect objects using specified methods
        
        Args:
            image: Input image
            methods: List of detection methods to use
            
        Returns:
            List of detected objects
        """
        all_detections = []
        
        if methods is None:
            methods = list(self.detection_methods.keys())
        
        for method_name in methods:
            if method_name not in self.detection_methods:
                continue
            
            method_config = self.detection_methods[method_name]
            method = method_config['method']
            params = method_config['params']
            
            try:
                if method == 'color':
                    detections = self.color_detector.detect_color_objects(image, **params)
                elif method == 'template':
                    detections = self.template_detector.detect_template(image, **params)
                elif method == 'contour':
                    detections = self.contour_detector.detect_objects(image, **params)
                else:
                    continue
                
                all_detections.extend(detections)
                
            except Exception as e:
                logging.error(f"Error in detection method {method_name}: {e}")
        
        return all_detections
    
    def track_objects(self, detected_objects: List[DetectedObject]) -> Dict[int, TrackedObject]:
        """Track detected objects"""
        if self.tracking_enabled:
            return self.multi_tracker.update(detected_objects)
        else:
            return {}
    
    def visualize_detections(self, image: np.ndarray, 
                           detected_objects: List[DetectedObject],
                           tracked_objects: Optional[Dict[int, TrackedObject]] = None) -> np.ndarray:
        """
        Visualize detections and tracking on image
        
        Args:
            image: Input image
            detected_objects: List of detected objects
            tracked_objects: Dictionary of tracked objects
            
        Returns:
            Annotated image
        """
        result_image = image.copy()
        
        # Draw detections
        for obj in detected_objects:
            x, y, w, h = obj.bbox
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center
            center = (int(obj.center[0]), int(obj.center[1]))
            cv2.circle(result_image, center, 3, (0, 255, 0), -1)
            
            # Draw label
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw tracking
        if tracked_objects:
            for track_id, tracked_obj in tracked_objects.items():
                if tracked_obj.state == TrackingState.ACTIVE:
                    # Draw trajectory
                    if len(tracked_obj.position_history) > 1:
                        points = [(int(p[0]), int(p[1])) for p in tracked_obj.position_history]
                        for i in range(1, len(points)):
                            cv2.line(result_image, points[i-1], points[i], (255, 0, 0), 2)
                    
                    # Draw ID
                    last_pos = tracked_obj.position_history[-1]
                    cv2.putText(result_image, f"ID: {track_id}", 
                               (int(last_pos[0]), int(last_pos[1]) + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return result_image