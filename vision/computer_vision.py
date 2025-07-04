#!/usr/bin/env python3
"""
Computer Vision Module for Robotics
Implements various computer vision algorithms for robot perception
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ObjectDetector:
    """
    Object detection using various methods
    """
    def __init__(self, method: str = 'contour'):
        self.method = method
        self.background_subtractor = None
        
    def detect_objects_contour(self, image: np.ndarray, 
                              min_area: int = 500) -> List[Dict[str, Any]]:
        """
        Detect objects using contour detection
        Args:
            image: Input image (H, W, 3)
            min_area: Minimum contour area
        Returns:
            List of detected objects with bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                objects.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'area': area,
                    'contour': contour
                })
        
        return objects
    
    def detect_objects_hog(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using HOG features
        Args:
            image: Input image (H, W, 3)
        Returns:
            List of detected objects
        """
        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect objects
        boxes, weights = hog.detectMultiScale(image, winStride=(8, 8))
        
        objects = []
        for i, (x, y, w, h) in enumerate(boxes):
            objects.append({
                'id': i,
                'bbox': (x, y, w, h),
                'centroid': (x + w//2, y + h//2),
                'confidence': weights[i] if i < len(weights) else 0.0,
                'type': 'person'
            })
        
        return objects
    
    def detect_objects_background_subtraction(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect moving objects using background subtraction
        Args:
            image: Input image (H, W, 3)
        Returns:
            List of detected moving objects
        """
        if self.background_subtractor is None:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(image)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'centroid': (x + w//2, y + h//2),
                    'area': area,
                    'type': 'moving_object'
                })
        
        return objects


class FeatureExtractor:
    """
    Feature extraction for computer vision
    """
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        
    def extract_sift_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SIFT features from image
        Args:
            image: Input image (H, W, 3)
        Returns:
            Keypoints and descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Convert keypoints to array
        kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        return kp_array, descriptors
    
    def extract_orb_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ORB features from image
        Args:
            image: Input image (H, W, 3)
        Returns:
            Keypoints and descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Convert keypoints to array
        kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        return kp_array, descriptors
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from image
        Args:
            image: Input image (H, W, 3)
        Returns:
            HOG feature vector
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        gray = cv2.resize(gray, (64, 128))
        
        # Compute HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        
        return features.flatten()


class OpticalFlow:
    """
    Optical flow estimation for motion tracking
    """
    def __init__(self, method: str = 'lucas_kanade'):
        self.method = method
        self.prev_frame = None
        self.tracks = []
        
    def lucas_kanade_optical_flow(self, current_frame: np.ndarray, 
                                 points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Lucas-Kanade optical flow
        Args:
            current_frame: Current frame (H, W, 3)
            points: Points to track (N, 2)
        Returns:
            Flow vectors (N, 2)
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.array([])
        
        if points is None:
            # Detect corners to track
            corners = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=100, 
                                            qualityLevel=0.3, minDistance=7, blockSize=7)
            if corners is not None:
                points = corners.reshape(-1, 2)
            else:
                return np.array([])
        
        # Calculate optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, 
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, points.astype(np.float32), None, **lk_params)
        
        # Select good points
        good_new = next_points[status == 1]
        good_old = points[status == 1]
        
        # Calculate flow vectors
        flow = good_new - good_old
        
        self.prev_frame = gray
        
        return flow
    
    def dense_optical_flow(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Dense optical flow using Farneback method
        Args:
            current_frame: Current frame (H, W, 3)
        Returns:
            Dense flow field (H, W, 2)
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros((gray.shape[0], gray.shape[1], 2))
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, None, None)
        
        self.prev_frame = gray
        
        return flow


class DepthEstimation:
    """
    Depth estimation from stereo vision or monocular cues
    """
    def __init__(self, method: str = 'stereo'):
        self.method = method
        self.stereo_matcher = cv2.StereoBM_create()
        
    def stereo_depth_estimation(self, left_image: np.ndarray, 
                               right_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from stereo image pair
        Args:
            left_image: Left stereo image (H, W, 3)
            right_image: Right stereo image (H, W, 3)
        Returns:
            Depth map (H, W)
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert to depth (assuming calibrated cameras)
        # depth = (focal_length * baseline) / disparity
        # For now, return normalized disparity
        depth = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        
        return depth
    
    def monocular_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from single image using geometric cues
        Args:
            image: Input image (H, W, 3)
        Returns:
            Estimated depth map (H, W)
        """
        # Simple depth estimation based on texture and gradient
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Use gradient magnitude as depth cue (higher gradient = closer)
        depth = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return depth.astype(np.uint8)


class SemanticSegmentation:
    """
    Semantic segmentation for scene understanding
    """
    def __init__(self, model_type: str = 'simple'):
        self.model_type = model_type
        
    def color_based_segmentation(self, image: np.ndarray, 
                                color_ranges: Dict[str, Tuple[Tuple, Tuple]]) -> np.ndarray:
        """
        Segment image based on color ranges
        Args:
            image: Input image (H, W, 3) in BGR
            color_ranges: Dictionary of color ranges {class_name: ((lower_hsv), (upper_hsv))}
        Returns:
            Segmentation mask (H, W)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Initialize segmentation mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Apply color ranges
        for class_id, (class_name, (lower, upper)) in enumerate(color_ranges.items()):
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask[color_mask > 0] = class_id + 1
        
        return mask
    
    def watershed_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Watershed segmentation for object separation
        Args:
            image: Input image (H, W, 3)
        Returns:
            Segmentation mask (H, W)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        return markers


class VisualSLAM:
    """
    Visual SLAM implementation
    """
    def __init__(self):
        self.keyframes = []
        self.map_points = []
        self.feature_extractor = FeatureExtractor()
        
    def process_frame(self, image: np.ndarray, camera_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for SLAM
        Args:
            image: Input image (H, W, 3)
            camera_matrix: Camera intrinsic matrix (3, 3)
        Returns:
            Dictionary with pose and map information
        """
        # Extract features
        keypoints, descriptors = self.feature_extractor.extract_sift_features(image)
        
        if len(self.keyframes) == 0:
            # First frame - initialize
            keyframe = {
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'pose': np.eye(4)
            }
            self.keyframes.append(keyframe)
            
            return {
                'pose': np.eye(4),
                'num_features': len(keypoints),
                'num_matches': 0
            }
        
        # Match features with previous keyframe
        matches = self.match_features(descriptors, self.keyframes[-1]['descriptors'])
        
        if len(matches) < 10:
            return {
                'pose': self.keyframes[-1]['pose'],
                'num_features': len(keypoints),
                'num_matches': len(matches)
            }
        
        # Estimate pose
        pose = self.estimate_pose(keypoints, self.keyframes[-1]['keypoints'], 
                                 matches, camera_matrix)
        
        # Add new keyframe
        keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose
        }
        self.keyframes.append(keyframe)
        
        return {
            'pose': pose,
            'num_features': len(keypoints),
            'num_matches': len(matches)
        }
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[Tuple[int, int]]:
        """
        Match features between two frames
        Args:
            desc1: Descriptors from frame 1
            desc2: Descriptors from frame 2
        Returns:
            List of matches (idx1, idx2)
        """
        if desc1 is None or desc2 is None:
            return []
        
        # Use FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))
        
        return good_matches
    
    def estimate_pose(self, kp1: np.ndarray, kp2: np.ndarray, 
                     matches: List[Tuple[int, int]], camera_matrix: np.ndarray) -> np.ndarray:
        """
        Estimate pose between two frames
        Args:
            kp1: Keypoints from frame 1
            kp2: Keypoints from frame 2
            matches: Feature matches
            camera_matrix: Camera intrinsic matrix
        Returns:
            Pose transformation matrix (4, 4)
        """
        if len(matches) < 5:
            return np.eye(4)
        
        # Extract matched points
        pts1 = np.float32([kp1[m[0]] for m in matches])
        pts2 = np.float32([kp2[m[1]] for m in matches])
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
        
        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)
        
        # Construct transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        return pose


class VisionProcessor:
    """
    Main vision processing pipeline
    """
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.feature_extractor = FeatureExtractor()
        self.optical_flow = OpticalFlow()
        self.depth_estimator = DepthEstimation()
        self.segmentation = SemanticSegmentation()
        self.slam = VisualSLAM()
        
    def process_frame(self, image: np.ndarray, camera_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Complete vision processing pipeline
        Args:
            image: Input image (H, W, 3)
            camera_matrix: Camera intrinsic matrix (3, 3)
        Returns:
            Dictionary with all vision results
        """
        results = {}
        
        # Object detection
        results['objects'] = self.object_detector.detect_objects_contour(image)
        
        # Feature extraction
        kp, desc = self.feature_extractor.extract_sift_features(image)
        results['features'] = {'keypoints': kp, 'descriptors': desc}
        
        # Optical flow
        flow = self.optical_flow.lucas_kanade_optical_flow(image)
        results['optical_flow'] = flow
        
        # Depth estimation (monocular)
        depth = self.depth_estimator.monocular_depth_estimation(image)
        results['depth'] = depth
        
        # Semantic segmentation
        color_ranges = {
            'red': ((0, 120, 70), (10, 255, 255)),
            'green': ((40, 40, 40), (80, 255, 255)),
            'blue': ((100, 60, 60), (140, 255, 255))
        }
        segmentation_mask = self.segmentation.color_based_segmentation(image, color_ranges)
        results['segmentation'] = segmentation_mask
        
        # SLAM
        slam_results = self.slam.process_frame(image, camera_matrix)
        results['slam'] = slam_results
        
        return results