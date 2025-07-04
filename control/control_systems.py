#!/usr/bin/env python3
"""
Control Systems Module for Robotics
Implements various control algorithms for robot systems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Callable
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_are, inv
import control as ct


class PIDController:
    """
    PID Controller implementation
    """
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Optional[Tuple[float, float]] = None,
                 windup_guard: Optional[float] = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.windup_guard = windup_guard
        
        # State variables
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def update(self, setpoint: float, measurement: float, dt: Optional[float] = None) -> float:
        """
        Update PID controller
        Args:
            setpoint: Desired value
            measurement: Current measurement
            dt: Time step (if None, use internal timing)
        Returns:
            Control output
        """
        # Calculate error
        error = setpoint - measurement
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        if dt is not None:
            self.integral += error * dt
        else:
            self.integral += error
            
        # Apply windup guard
        if self.windup_guard is not None:
            self.integral = np.clip(self.integral, -self.windup_guard, self.windup_guard)
        
        integral_term = self.ki * self.integral
        
        # Derivative term
        if dt is not None:
            derivative = (error - self.last_error) / dt if dt > 0 else 0
        else:
            derivative = error - self.last_error
            
        derivative_term = self.kd * derivative
        
        # Calculate output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.last_error = error
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None


class LQRController:
    """
    Linear Quadratic Regulator (LQR) Controller
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Solve Riccati equation
        self.P = solve_continuous_are(A, B, Q, R)
        
        # Calculate feedback gain
        self.K = inv(R) @ B.T @ self.P
        
    def control(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Calculate LQR control input
        Args:
            state: Current state vector
            reference: Reference state vector
        Returns:
            Control input vector
        """
        error = reference - state
        return self.K @ error


class MPCController:
    """
    Model Predictive Controller (MPC)
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 prediction_horizon: int, control_horizon: int,
                 state_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 input_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = prediction_horizon
        self.M = control_horizon
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints
        
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        
    def predict_trajectory(self, x0: np.ndarray, u_sequence: np.ndarray) -> np.ndarray:
        """
        Predict system trajectory given initial state and input sequence
        Args:
            x0: Initial state
            u_sequence: Input sequence (M, n_inputs)
        Returns:
            Predicted state trajectory (N+1, n_states)
        """
        trajectory = np.zeros((self.N + 1, self.n_states))
        trajectory[0] = x0
        
        for k in range(self.N):
            if k < len(u_sequence):
                u = u_sequence[k]
            else:
                u = np.zeros(self.n_inputs)
            
            trajectory[k + 1] = self.A @ trajectory[k] + self.B @ u
        
        return trajectory
    
    def cost_function(self, u_sequence: np.ndarray, x0: np.ndarray, 
                     reference: np.ndarray) -> float:
        """
        Calculate MPC cost function
        Args:
            u_sequence: Input sequence (M * n_inputs,)
            x0: Initial state
            reference: Reference trajectory (N+1, n_states)
        Returns:
            Total cost
        """
        # Reshape input sequence
        u_seq = u_sequence.reshape((self.M, self.n_inputs))
        
        # Predict trajectory
        trajectory = self.predict_trajectory(x0, u_seq)
        
        # Calculate cost
        cost = 0.0
        
        # State cost
        for k in range(self.N + 1):
            if k < len(reference):
                ref = reference[k]
            else:
                ref = reference[-1]
            
            state_error = trajectory[k] - ref
            cost += state_error.T @ self.Q @ state_error
        
        # Input cost
        for k in range(self.M):
            cost += u_seq[k].T @ self.R @ u_seq[k]
        
        return cost
    
    def control(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Calculate MPC control input
        Args:
            state: Current state
            reference: Reference trajectory
        Returns:
            Control input
        """
        # Initial guess for optimization
        u0 = np.zeros(self.M * self.n_inputs)
        
        # Constraints
        constraints = []
        bounds = []
        
        # Input constraints
        if self.input_constraints is not None:
            u_min, u_max = self.input_constraints
            for i in range(self.M):
                for j in range(self.n_inputs):
                    idx = i * self.n_inputs + j
                    bounds.append((u_min[j], u_max[j]))
        else:
            bounds = [(-np.inf, np.inf)] * (self.M * self.n_inputs)
        
        # State constraints (implemented as nonlinear constraints)
        if self.state_constraints is not None:
            def state_constraint(u_sequence):
                u_seq = u_sequence.reshape((self.M, self.n_inputs))
                trajectory = self.predict_trajectory(state, u_seq)
                
                violations = []
                x_min, x_max = self.state_constraints
                
                for k in range(1, self.N + 1):
                    for j in range(self.n_states):
                        violations.append(trajectory[k, j] - x_min[j])  # x >= x_min
                        violations.append(x_max[j] - trajectory[k, j])  # x <= x_max
                
                return np.array(violations)
            
            constraints.append({
                'type': 'ineq',
                'fun': state_constraint
            })
        
        # Solve optimization problem
        result = minimize(
            self.cost_function,
            u0,
            args=(state, reference),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            u_optimal = result.x.reshape((self.M, self.n_inputs))
            return u_optimal[0]  # Return first control input
        else:
            print("MPC optimization failed")
            return np.zeros(self.n_inputs)


class AdaptiveController:
    """
    Adaptive controller using parameter estimation
    """
    def __init__(self, n_params: int, adaptation_rate: float = 0.1):
        self.n_params = n_params
        self.adaptation_rate = adaptation_rate
        self.theta = np.zeros(n_params)  # Parameter estimates
        self.P = np.eye(n_params) * 1000  # Covariance matrix
        
    def update_parameters(self, phi: np.ndarray, error: float):
        """
        Update parameter estimates using recursive least squares
        Args:
            phi: Regressor vector
            error: Prediction error
        """
        # RLS update
        K = self.P @ phi / (1 + phi.T @ self.P @ phi)
        self.theta += K * error
        self.P = self.P - np.outer(K, phi.T @ self.P)
    
    def control(self, phi: np.ndarray, reference: float, measurement: float) -> float:
        """
        Calculate adaptive control input
        Args:
            phi: Regressor vector
            reference: Reference signal
            measurement: Current measurement
        Returns:
            Control input
        """
        # Prediction
        prediction = phi.T @ self.theta
        
        # Error
        error = measurement - prediction
        
        # Update parameters
        self.update_parameters(phi, error)
        
        # Calculate control (simplified adaptive law)
        control_input = reference - prediction
        
        return control_input


class SlidingModeController:
    """
    Sliding Mode Controller for robust control
    """
    def __init__(self, c: float, eta: float, boundary_layer: float = 0.1):
        self.c = c  # Sliding surface parameter
        self.eta = eta  # Switching gain
        self.boundary_layer = boundary_layer
        
    def sliding_surface(self, error: float, error_dot: float) -> float:
        """
        Calculate sliding surface value
        Args:
            error: Tracking error
            error_dot: Derivative of tracking error
        Returns:
            Sliding surface value
        """
        return error_dot + self.c * error
    
    def control(self, error: float, error_dot: float) -> float:
        """
        Calculate sliding mode control input
        Args:
            error: Tracking error
            error_dot: Derivative of tracking error
        Returns:
            Control input
        """
        s = self.sliding_surface(error, error_dot)
        
        # Smooth switching function (to avoid chattering)
        if abs(s) <= self.boundary_layer:
            switching_term = s / self.boundary_layer
        else:
            switching_term = np.sign(s)
        
        return -self.eta * switching_term


class ImpedanceController:
    """
    Impedance controller for robot manipulation
    """
    def __init__(self, M_d: np.ndarray, C_d: np.ndarray, K_d: np.ndarray):
        self.M_d = M_d  # Desired inertia matrix
        self.C_d = C_d  # Desired damping matrix
        self.K_d = K_d  # Desired stiffness matrix
        
    def control(self, position: np.ndarray, velocity: np.ndarray,
               position_ref: np.ndarray, velocity_ref: np.ndarray,
               acceleration_ref: np.ndarray, force_external: np.ndarray,
               M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Calculate impedance control torques
        Args:
            position: Current position
            velocity: Current velocity
            position_ref: Reference position
            velocity_ref: Reference velocity
            acceleration_ref: Reference acceleration
            force_external: External force
            M: Robot inertia matrix
            C: Robot Coriolis/centripetal matrix
        Returns:
            Control torques
        """
        # Position and velocity errors
        e_p = position_ref - position
        e_v = velocity_ref - velocity
        
        # Desired acceleration
        a_d = acceleration_ref + self.M_d @ (self.K_d @ e_p + self.C_d @ e_v) + force_external
        
        # Control law
        tau = M @ a_d + C @ velocity
        
        return tau


class VisualServoController:
    """
    Visual servoing controller
    """
    def __init__(self, lambda_gain: float = 0.1):
        self.lambda_gain = lambda_gain
        
    def image_jacobian(self, features: np.ndarray, depth: np.ndarray, 
                      camera_params: Dict[str, float]) -> np.ndarray:
        """
        Calculate image Jacobian matrix
        Args:
            features: Image features (N, 2) in pixel coordinates
            depth: Depth of features (N,)
            camera_params: Camera parameters {'fx', 'fy', 'cx', 'cy'}
        Returns:
            Image Jacobian (2N, 6)
        """
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        n_features = len(features)
        L = np.zeros((2 * n_features, 6))
        
        for i, (x, y) in enumerate(features):
            # Convert to normalized coordinates
            x_norm = (x - cx) / fx
            y_norm = (y - cy) / fy
            Z = depth[i]
            
            # Image Jacobian for point i
            L[2*i] = [-1/Z, 0, x_norm/Z, x_norm*y_norm, -(1+x_norm**2), y_norm]
            L[2*i+1] = [0, -1/Z, y_norm/Z, 1+y_norm**2, -x_norm*y_norm, -x_norm]
        
        return L
    
    def control(self, current_features: np.ndarray, desired_features: np.ndarray,
               depth: np.ndarray, camera_params: Dict[str, float]) -> np.ndarray:
        """
        Calculate visual servoing control velocities
        Args:
            current_features: Current image features (N, 2)
            desired_features: Desired image features (N, 2)
            depth: Feature depths (N,)
            camera_params: Camera parameters
        Returns:
            Camera velocity commands (6,) [vx, vy, vz, wx, wy, wz]
        """
        # Feature error
        error = (desired_features - current_features).flatten()
        
        # Image Jacobian
        L = self.image_jacobian(current_features, depth, camera_params)
        
        # Pseudo-inverse control law
        L_pinv = np.linalg.pinv(L)
        velocity = -self.lambda_gain * L_pinv @ error
        
        return velocity


class ForceController:
    """
    Force controller for contact tasks
    """
    def __init__(self, kp_force: float = 1.0, ki_force: float = 0.1):
        self.kp_force = kp_force
        self.ki_force = ki_force
        self.force_integral = 0.0
        
    def hybrid_position_force_control(self, position: np.ndarray, force: np.ndarray,
                                    position_ref: np.ndarray, force_ref: np.ndarray,
                                    selection_matrix: np.ndarray, dt: float) -> np.ndarray:
        """
        Hybrid position/force control
        Args:
            position: Current position
            force: Current force measurement
            position_ref: Reference position
            force_ref: Reference force
            selection_matrix: Selection matrix (1 for force control, 0 for position)
            dt: Time step
        Returns:
            Control command
        """
        # Position error
        pos_error = position_ref - position
        
        # Force error
        force_error = force_ref - force
        self.force_integral += force_error * dt
        
        # Position control
        pos_control = pos_error  # Simplified P control
        
        # Force control
        force_control = self.kp_force * force_error + self.ki_force * self.force_integral
        
        # Hybrid control
        control = (np.eye(len(position)) - selection_matrix) @ pos_control + selection_matrix @ force_control
        
        return control


class TrajectoryTracker:
    """
    Trajectory tracking controller
    """
    def __init__(self, controller_type: str = 'pd'):
        self.controller_type = controller_type
        
    def pd_tracking(self, position: np.ndarray, velocity: np.ndarray,
                   position_ref: np.ndarray, velocity_ref: np.ndarray,
                   kp: float, kd: float) -> np.ndarray:
        """
        PD trajectory tracking
        Args:
            position: Current position
            velocity: Current velocity
            position_ref: Reference position
            velocity_ref: Reference velocity
            kp: Proportional gain
            kd: Derivative gain
        Returns:
            Control torques
        """
        pos_error = position_ref - position
        vel_error = velocity_ref - velocity
        
        return kp * pos_error + kd * vel_error
    
    def computed_torque_control(self, position: np.ndarray, velocity: np.ndarray,
                               position_ref: np.ndarray, velocity_ref: np.ndarray,
                               acceleration_ref: np.ndarray, M: np.ndarray,
                               C: np.ndarray, G: np.ndarray,
                               kp: float, kd: float) -> np.ndarray:
        """
        Computed torque control for trajectory tracking
        Args:
            position: Current position
            velocity: Current velocity
            position_ref: Reference position
            velocity_ref: Reference velocity
            acceleration_ref: Reference acceleration
            M: Inertia matrix
            C: Coriolis matrix
            G: Gravity vector
            kp: Proportional gain
            kd: Derivative gain
        Returns:
            Control torques
        """
        pos_error = position_ref - position
        vel_error = velocity_ref - velocity
        
        # Feedback linearization
        a_d = acceleration_ref + kp * pos_error + kd * vel_error
        
        # Computed torque
        tau = M @ a_d + C @ velocity + G
        
        return tau


class RobustController:
    """
    Robust controller with uncertainty handling
    """
    def __init__(self, nominal_params: Dict[str, float], uncertainty_bounds: Dict[str, float]):
        self.nominal_params = nominal_params
        self.uncertainty_bounds = uncertainty_bounds
        
    def h_infinity_control(self, state: np.ndarray, reference: np.ndarray,
                          A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        """
        H-infinity robust control
        Args:
            state: Current state
            reference: Reference state
            A: System matrix
            B: Input matrix
            gamma: H-infinity bound
        Returns:
            Robust control input
        """
        # Simplified H-infinity control (requires more sophisticated design)
        error = reference - state
        
        # For demonstration, use LQR-like gain with robustness margin
        Q = np.eye(len(state))
        R = np.eye(B.shape[1]) * gamma
        
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K @ error
        except:
            return np.zeros(B.shape[1])


# Control System Analysis and Design Tools
class ControlAnalysis:
    """
    Tools for control system analysis and design
    """
    @staticmethod
    def stability_analysis(A: np.ndarray) -> Dict[str, Any]:
        """
        Analyze system stability
        Args:
            A: System matrix
        Returns:
            Stability analysis results
        """
        eigenvalues = np.linalg.eigvals(A)
        
        # Check stability
        stable = np.all(np.real(eigenvalues) < 0)
        
        # Dominant pole
        dominant_pole = eigenvalues[np.argmax(np.real(eigenvalues))]
        
        return {
            'eigenvalues': eigenvalues,
            'stable': stable,
            'dominant_pole': dominant_pole,
            'settling_time': 4 / abs(np.real(dominant_pole)) if np.real(dominant_pole) < 0 else np.inf
        }
    
    @staticmethod
    def controllability_analysis(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
        """
        Analyze system controllability
        Args:
            A: System matrix
            B: Input matrix
        Returns:
            Controllability analysis results
        """
        n = A.shape[0]
        
        # Controllability matrix
        C_ctrl = B
        for i in range(1, n):
            C_ctrl = np.hstack([C_ctrl, np.linalg.matrix_power(A, i) @ B])
        
        # Check controllability
        rank = np.linalg.matrix_rank(C_ctrl)
        controllable = rank == n
        
        return {
            'controllability_matrix': C_ctrl,
            'rank': rank,
            'controllable': controllable
        }
    
    @staticmethod
    def observability_analysis(A: np.ndarray, C: np.ndarray) -> Dict[str, Any]:
        """
        Analyze system observability
        Args:
            A: System matrix
            C: Output matrix
        Returns:
            Observability analysis results
        """
        n = A.shape[0]
        
        # Observability matrix
        O_obs = C
        for i in range(1, n):
            O_obs = np.vstack([O_obs, C @ np.linalg.matrix_power(A, i)])
        
        # Check observability
        rank = np.linalg.matrix_rank(O_obs)
        observable = rank == n
        
        return {
            'observability_matrix': O_obs,
            'rank': rank,
            'observable': observable
        }