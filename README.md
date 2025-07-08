# Robotics Perception and Control

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)](https://opencv.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic-blue.svg)](https://www.ros.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced robotics perception and control systems featuring computer vision algorithms, sensor fusion, and control theory implementations. Designed for autonomous robots, industrial automation, and research applications. This repository presents a structured collection of code and assignments completed for ESE 6500 spring 2025ver at the University of Pennsylvania.

## 🎯 Features

- **Computer Vision**: Object detection, tracking, and recognition
- **Sensor Fusion**: Multi-modal sensor integration
- **Control Systems**: PID, LQR, and model predictive control
- **Path Planning**: RRT, A*, and trajectory optimization
- **Real-time Processing**: Optimized for robotic applications
- **ROS Integration**: Compatible with Robot Operating System

## 🚀 Quick Start

```python
from src.perception import VisionSystem
from src.control import RobotController

# Initialize vision system
vision = VisionSystem()
vision.load_camera_config('configs/camera_calibration.yaml')

# Object detection and tracking
objects = vision.detect_objects(camera_frame)
tracked_objects = vision.track_objects(objects)

# Control system
controller = RobotController()
controller.set_target(tracked_objects[0].position)
control_commands = controller.compute_control()
```

## 🔍 Perception Modules

### Computer Vision
- **Object Detection**: YOLO, SSD, and custom detectors
- **Feature Tracking**: ORB, SIFT, and optical flow
- **Depth Estimation**: Stereo vision and structured light
- **Semantic Segmentation**: Real-time scene understanding

### Sensor Fusion
- **Multi-Camera Systems**: Synchronized capture and processing
- **LiDAR Integration**: Point cloud processing and registration
- **IMU Fusion**: Attitude and motion estimation
- **GPS Integration**: Global localization

## 🎮 Control Systems

### Classical Control
```python
# PID Controller
pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
output = pid.update(setpoint, current_value, dt)

# LQR Controller  
lqr = LQRController(A, B, Q, R)
u = lqr.compute_control(x_current, x_target)
```

### Advanced Control
```python
# Model Predictive Control
mpc = MPCController(
    prediction_horizon=10,
    control_horizon=5,
    state_constraints=state_bounds,
    input_constraints=input_bounds
)
optimal_control = mpc.solve(current_state, reference_trajectory)
```

## 📁 Project Structure

```
robotics-perception-control/
├── src/                        # Source code
│   ├── perception/             # Vision and sensing
│   │   ├── camera_system.py    # Camera interface
│   │   ├── object_detector.py  # Object detection
│   │   └── sensor_fusion.py    # Multi-sensor fusion
│   ├── control/                # Control algorithms
│   │   ├── pid_controller.py   # PID implementation
│   │   ├── lqr_controller.py   # Linear quadratic regulator
│   │   └── mpc_controller.py   # Model predictive control
│   └── planning/               # Path planning
│       ├── rrt_planner.py      # Rapidly-exploring random trees
│       └── astar_planner.py    # A* path planning
├── examples/                   # Example applications
├── configs/                    # Configuration files
├── tests/                      # Test suite
├── docs/                       # Documentation
└── README.md                  # This file
```

## 🛠 Hardware Integration

### Supported Hardware
- **Cameras**: USB, GigE, and IP cameras
- **LiDAR**: Velodyne, Ouster, and SICK sensors
- **IMU**: MPU-9250, BNO055, and industrial IMUs
- **Motors**: Servo, stepper, and brushless DC motors

### Communication Protocols
- **ROS/ROS2**: Standard robotics middleware
- **Ethernet**: High-speed data transmission
- **Serial**: Arduino and microcontroller integration
- **CAN Bus**: Industrial automation protocols

## 🎯 Applications

### Autonomous Vehicles
```python
# Autonomous driving pipeline
pipeline = AutonomousDrivingPipeline()
pipeline.add_module('lane_detection', LaneDetector())
pipeline.add_module('obstacle_detection', ObstacleDetector())
pipeline.add_module('path_planner', PathPlanner())
pipeline.add_module('vehicle_controller', VehicleController())

# Process driving decision
driving_command = pipeline.process(sensor_data)
```

### Industrial Automation
```python
# Robotic arm control
arm_controller = RoboticArmController(dof=6)
arm_controller.load_kinematics('configs/ur5_kinematics.yaml')

# Pick and place operation
pick_pose = vision.detect_object_pose('target_object')
place_pose = Pose(x=0.5, y=0.3, z=0.2)

trajectory = arm_controller.plan_trajectory(pick_pose, place_pose)
arm_controller.execute_trajectory(trajectory)
```

### Mobile Robotics
```python
# Navigation system
navigator = MobileRobotNavigator()
navigator.set_map('maps/office_environment.pgm')

# Autonomous navigation
goal_pose = Pose(x=5.0, y=3.0, theta=1.57)
path = navigator.plan_path(current_pose, goal_pose)
navigator.execute_path(path)
```

## 📊 Performance Metrics

Performance will vary based on your specific hardware, algorithms, and application requirements. The system is designed to provide real-time processing capabilities for robotics applications.

## 🔧 Advanced Features

### Machine Learning Integration
```python
# Deep learning for perception
detector = DeepObjectDetector('models/yolov5_custom.pt')
classifier = ImageClassifier('models/resnet50_finetuned.pth')

# Reinforcement learning for control
rl_controller = RLController('models/ppo_navigation.zip')
action = rl_controller.predict(observation)
```

### Real-time Optimization
```python
# Real-time trajectory optimization
optimizer = TrajectoryOptimizer()
optimal_trajectory = optimizer.optimize(
    start_state=current_state,
    goal_state=target_state,
    constraints=environment_constraints,
    time_horizon=5.0
)
```

## 🔬 Research Applications

- **Academic Research**: Robotics algorithms development
- **Industry**: Manufacturing automation and quality control
- **Defense**: Autonomous systems and surveillance
- **Healthcare**: Surgical robotics and rehabilitation

## 🤝 Contributing

We welcome contributions in perception algorithms, control methods, and hardware integration!

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

👁️ **See, Think, Act - Complete Robotics Intelligence**
