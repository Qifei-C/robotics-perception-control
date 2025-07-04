"""
Camera System Module
Provides camera interfaces and calibration for robotics vision systems
"""

import cv2
import numpy as np
import yaml
import json
from typing import Dict, List, Tuple, Optional, Union
import threading
import time
from datetime import datetime
import logging


class CameraCalibration:
    """
    Camera calibration utilities
    """
    
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.image_size = None
        self.calibration_error = None
        
    def calibrate_camera(self, calibration_images: List[np.ndarray], 
                        chessboard_size: Tuple[int, int] = (9, 6),
                        square_size: float = 25.0) -> Dict:
        """
        Calibrate camera using chessboard pattern
        
        Args:
            calibration_images: List of calibration images
            chessboard_size: Number of inner corners (width, height)
            square_size: Size of chessboard squares in mm
            
        Returns:
            Calibration results dictionary
        """
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if len(objpoints) == 0:
            raise ValueError("No chessboard patterns found in calibration images")
        
        # Perform camera calibration
        self.image_size = gray.shape[::-1]
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = dist_coeffs
        self.calibration_error = ret
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                            camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        return {
            'camera_matrix': camera_matrix,
            'distortion_coeffs': dist_coeffs,
            'image_size': self.image_size,
            'calibration_error': ret,
            'mean_reprojection_error': mean_error,
            'num_images_used': len(objpoints)
        }
    
    def save_calibration(self, filepath: str, calibration_data: Dict):
        """Save calibration data to file"""
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {
            'camera_matrix': calibration_data['camera_matrix'].tolist(),
            'distortion_coeffs': calibration_data['distortion_coeffs'].tolist(),
            'image_size': calibration_data['image_size'],
            'calibration_error': float(calibration_data['calibration_error']),
            'mean_reprojection_error': float(calibration_data['mean_reprojection_error']),
            'num_images_used': calibration_data['num_images_used'],
            'calibration_date': datetime.now().isoformat()
        }
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(data_to_save, f, default_flow_style=False)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
    
    def load_calibration(self, filepath: str) -> Dict:
        """Load calibration data from file"""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
        
        # Convert back to numpy arrays
        self.camera_matrix = np.array(data['camera_matrix'])
        self.distortion_coeffs = np.array(data['distortion_coeffs'])
        self.image_size = tuple(data['image_size'])
        self.calibration_error = data['calibration_error']
        
        return data
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using calibration parameters"""
        if self.camera_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Camera not calibrated. Load calibration data first.")
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)


class CameraInterface:
    """
    Generic camera interface for different camera types
    """
    
    def __init__(self, camera_id: Union[int, str] = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_connected = False
        self.frame_rate = 30
        self.resolution = (640, 480)
        self.calibration = CameraCalibration()
        
    def connect(self) -> bool:
        """Connect to camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                self.is_connected = True
                self.set_resolution(self.resolution)
                self.set_frame_rate(self.frame_rate)
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Failed to connect to camera {self.camera_id}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False
    
    def set_resolution(self, resolution: Tuple[int, int]):
        """Set camera resolution"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.resolution = resolution
    
    def set_frame_rate(self, fps: int):
        """Set camera frame rate"""
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.frame_rate = fps
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame"""
        if not self.is_connected or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
    
    def get_camera_info(self) -> Dict:
        """Get camera information"""
        if not self.is_connected:
            return {}
        
        return {
            'camera_id': self.camera_id,
            'resolution': self.resolution,
            'frame_rate': self.frame_rate,
            'backend': self.cap.get(cv2.CAP_PROP_BACKEND),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }


class StereoCamera:
    """
    Stereo camera system for depth estimation
    """
    
    def __init__(self, left_camera_id: Union[int, str] = 0, 
                 right_camera_id: Union[int, str] = 1):
        self.left_camera = CameraInterface(left_camera_id)
        self.right_camera = CameraInterface(right_camera_id)
        self.stereo_calibrated = False
        
        # Stereo calibration parameters
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.Q = None  # Disparity-to-depth mapping matrix
        
        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
    
    def connect(self) -> bool:
        """Connect both cameras"""
        left_ok = self.left_camera.connect()
        right_ok = self.right_camera.connect()
        return left_ok and right_ok
    
    def disconnect(self):
        """Disconnect both cameras"""
        self.left_camera.disconnect()
        self.right_camera.disconnect()
    
    def capture_stereo_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized stereo pair"""
        left_frame = self.left_camera.capture_frame()
        right_frame = self.right_camera.capture_frame()
        return left_frame, right_frame
    
    def calibrate_stereo(self, left_images: List[np.ndarray], 
                        right_images: List[np.ndarray],
                        chessboard_size: Tuple[int, int] = (9, 6),
                        square_size: float = 25.0) -> Dict:
        """
        Calibrate stereo camera system
        
        Args:
            left_images: Calibration images from left camera
            right_images: Calibration images from right camera
            chessboard_size: Chessboard pattern size
            square_size: Square size in mm
            
        Returns:
            Stereo calibration results
        """
        if len(left_images) != len(right_images):
            raise ValueError("Number of left and right images must match")
        
        # First calibrate individual cameras
        left_cal = self.left_camera.calibration.calibrate_camera(
            left_images, chessboard_size, square_size
        )
        right_cal = self.right_camera.calibration.calibrate_camera(
            right_images, chessboard_size, square_size
        )
        
        # Prepare object points for stereo calibration
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        
        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Find corners in both images
            ret_left, corners_left = cv2.findChessboardCorners(left_gray, chessboard_size)
            ret_right, corners_right = cv2.findChessboardCorners(right_gray, chessboard_size)
            
            if ret_left and ret_right:
                objpoints.append(objp)
                
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
                
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
        
        # Stereo calibration
        image_size = left_gray.shape[::-1]
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            left_cal['camera_matrix'], left_cal['distortion_coeffs'],
            right_cal['camera_matrix'], right_cal['distortion_coeffs'],
            image_size, flags=flags
        )
        
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_cal['camera_matrix'], left_cal['distortion_coeffs'],
            right_cal['camera_matrix'], right_cal['distortion_coeffs'],
            image_size, R, T
        )
        
        self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            left_cal['camera_matrix'], left_cal['distortion_coeffs'],
            R1, P1, image_size, cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            right_cal['camera_matrix'], right_cal['distortion_coeffs'],
            R2, P2, image_size, cv2.CV_16SC2
        )
        
        self.stereo_calibrated = True
        
        return {
            'left_calibration': left_cal,
            'right_calibration': right_cal,
            'stereo_error': ret,
            'rotation_matrix': R,
            'translation_vector': T,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'Q_matrix': Q,
            'baseline': np.linalg.norm(T)  # Distance between cameras
        }
    
    def rectify_stereo_pair(self, left_img: np.ndarray, 
                           right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify stereo image pair"""
        if not self.stereo_calibrated:
            raise ValueError("Stereo system not calibrated")
        
        left_rectified = cv2.remap(left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray,
                         num_disparities: int = 64, block_size: int = 15) -> np.ndarray:
        """Compute disparity map from stereo pair"""
        if not self.stereo_calibrated:
            raise ValueError("Stereo system not calibrated")
        
        # Rectify images
        left_rect, right_rect = self.rectify_stereo_pair(left_img, right_img)
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray)
        
        return disparity.astype(np.float32) / 16.0  # Convert to float
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Convert disparity map to depth map"""
        if self.Q is None:
            raise ValueError("Q matrix not available. Perform stereo calibration first.")
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Extract depth (Z coordinate)
        depth = points_3d[:, :, 2]
        
        # Filter invalid depths
        depth[depth <= 0] = 0
        depth[depth > 10000] = 0  # Remove very far points
        
        return depth


class MultiCameraSystem:
    """
    Multi-camera system for comprehensive scene coverage
    """
    
    def __init__(self):
        self.cameras = {}
        self.sync_enabled = False
        self.recording = False
        
    def add_camera(self, name: str, camera: CameraInterface):
        """Add camera to system"""
        self.cameras[name] = camera
        
    def remove_camera(self, name: str):
        """Remove camera from system"""
        if name in self.cameras:
            self.cameras[name].disconnect()
            del self.cameras[name]
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect all cameras"""
        results = {}
        for name, camera in self.cameras.items():
            results[name] = camera.connect()
        return results
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        for camera in self.cameras.values():
            camera.disconnect()
    
    def capture_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Capture frames from all cameras"""
        frames = {}
        for name, camera in self.cameras.items():
            frames[name] = camera.capture_frame()
        return frames
    
    def set_all_resolutions(self, resolution: Tuple[int, int]):
        """Set resolution for all cameras"""
        for camera in self.cameras.values():
            camera.set_resolution(resolution)
    
    def set_all_frame_rates(self, fps: int):
        """Set frame rate for all cameras"""
        for camera in self.cameras.values():
            camera.set_frame_rate(fps)
    
    def get_system_status(self) -> Dict:
        """Get status of all cameras"""
        status = {}
        for name, camera in self.cameras.items():
            status[name] = {
                'connected': camera.is_connected,
                'info': camera.get_camera_info() if camera.is_connected else {}
            }
        return status