"""
PID Controller Implementation
Classic Proportional-Integral-Derivative controller for robotics applications
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt


class PIDMode(Enum):
    """PID controller modes"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"


@dataclass
class PIDParameters:
    """PID controller parameters"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    output_min: float = -float('inf')  # Minimum output
    output_max: float = float('inf')   # Maximum output
    integral_min: float = -float('inf')  # Integral windup limits
    integral_max: float = float('inf')
    derivative_filter_alpha: float = 0.1  # Derivative filter coefficient


class PIDController:
    """
    PID Controller with advanced features
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 output_min: float = -float('inf'), output_max: float = float('inf'),
                 integral_windup_limit: Optional[float] = None,
                 derivative_filter: bool = True):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_min: Minimum output value
            output_max: Maximum output value
            integral_windup_limit: Integral windup protection limit
            derivative_filter: Enable derivative filtering
        """
        self.params = PIDParameters(
            kp=kp, ki=ki, kd=kd,
            output_min=output_min, output_max=output_max
        )
        
        if integral_windup_limit is not None:
            self.params.integral_min = -integral_windup_limit
            self.params.integral_max = integral_windup_limit
        
        # Internal state
        self.reset()
        
        # Configuration
        self.mode = PIDMode.AUTOMATIC
        self.derivative_filter_enabled = derivative_filter
        
        # History for analysis
        self.history = {
            'time': [],
            'setpoint': [],
            'process_variable': [],
            'error': [],
            'proportional': [],
            'integral': [],
            'derivative': [],
            'output': []
        }
        
    def reset(self):
        """Reset controller internal state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_derivative = 0.0
        self.last_time = None
        self.last_process_variable = None
        
    def set_parameters(self, kp: Optional[float] = None, 
                      ki: Optional[float] = None, 
                      kd: Optional[float] = None):
        """Update PID parameters"""
        if kp is not None:
            self.params.kp = kp
        if ki is not None:
            self.params.ki = ki
        if kd is not None:
            self.params.kd = kd
    
    def set_output_limits(self, min_output: float, max_output: float):
        """Set output limits"""
        self.params.output_min = min_output
        self.params.output_max = max_output
    
    def set_integral_limits(self, min_integral: float, max_integral: float):
        """Set integral windup limits"""
        self.params.integral_min = min_integral
        self.params.integral_max = max_integral
    
    def update(self, setpoint: float, process_variable: float, 
               dt: Optional[float] = None) -> float:
        """
        Update PID controller
        
        Args:
            setpoint: Desired value
            process_variable: Current measured value
            dt: Time step (if None, auto-calculated)
            
        Returns:
            Control output
        """
        if self.mode != PIDMode.AUTOMATIC:
            return 0.0
        
        current_time = time.time()
        
        # Calculate time step
        if dt is None:
            if self.last_time is None:
                dt = 0.01  # Default dt
            else:
                dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01  # Prevent division by zero
        
        # Calculate error
        error = setpoint - process_variable
        
        # Proportional term
        proportional = self.params.kp * error
        
        # Integral term
        self.integral += error * dt
        
        # Apply integral limits (windup protection)
        self.integral = np.clip(self.integral, 
                               self.params.integral_min, 
                               self.params.integral_max)
        
        integral_term = self.params.ki * self.integral
        
        # Derivative term
        if self.last_process_variable is not None:
            # Use derivative of process variable to avoid derivative kick
            derivative = -(process_variable - self.last_process_variable) / dt
        else:
            derivative = 0.0
        
        # Apply derivative filter
        if self.derivative_filter_enabled:
            alpha = self.params.derivative_filter_alpha
            derivative = alpha * derivative + (1 - alpha) * self.last_derivative
        
        derivative_term = self.params.kd * derivative
        
        # Calculate output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = np.clip(output, self.params.output_min, self.params.output_max)
        
        # Update state
        self.last_error = error
        self.last_derivative = derivative
        self.last_time = current_time
        self.last_process_variable = process_variable
        
        # Store history
        self.history['time'].append(current_time)
        self.history['setpoint'].append(setpoint)
        self.history['process_variable'].append(process_variable)
        self.history['error'].append(error)
        self.history['proportional'].append(proportional)
        self.history['integral'].append(integral_term)
        self.history['derivative'].append(derivative_term)
        self.history['output'].append(output)
        
        return output
    
    def set_mode(self, mode: PIDMode):
        """Set controller mode"""
        if mode == PIDMode.AUTOMATIC and self.mode == PIDMode.MANUAL:
            # Initialize for bumpless transfer
            self.reset()
        
        self.mode = mode
    
    def get_components(self) -> Dict[str, float]:
        """Get individual PID components"""
        if not self.history['time']:
            return {'proportional': 0, 'integral': 0, 'derivative': 0}
        
        return {
            'proportional': self.history['proportional'][-1],
            'integral': self.history['integral'][-1],
            'derivative': self.history['derivative'][-1]
        }
    
    def clear_history(self):
        """Clear controller history"""
        for key in self.history:
            self.history[key].clear()
    
    def plot_response(self, title: str = "PID Controller Response"):
        """Plot controller response"""
        if not self.history['time']:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot setpoint vs process variable
        axes[0].plot(self.history['time'], self.history['setpoint'], 
                    'r--', label='Setpoint', linewidth=2)
        axes[0].plot(self.history['time'], self.history['process_variable'], 
                    'b-', label='Process Variable', linewidth=2)
        axes[0].set_ylabel('Value')
        axes[0].set_title('Setpoint Tracking')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot PID components
        axes[1].plot(self.history['time'], self.history['proportional'], 
                    'g-', label='Proportional', linewidth=1)
        axes[1].plot(self.history['time'], self.history['integral'], 
                    'b-', label='Integral', linewidth=1)
        axes[1].plot(self.history['time'], self.history['derivative'], 
                    'r-', label='Derivative', linewidth=1)
        axes[1].set_ylabel('Component Value')
        axes[1].set_title('PID Components')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot control output
        axes[2].plot(self.history['time'], self.history['output'], 
                    'k-', label='Control Output', linewidth=2)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Output')
        axes[2].set_title('Control Output')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class MultiAxisPIDController:
    """
    Multi-axis PID controller for systems with multiple degrees of freedom
    """
    
    def __init__(self, num_axes: int):
        """
        Initialize multi-axis PID controller
        
        Args:
            num_axes: Number of control axes
        """
        self.num_axes = num_axes
        self.controllers = [PIDController() for _ in range(num_axes)]
        self.axis_names = [f"Axis_{i}" for i in range(num_axes)]
    
    def set_axis_parameters(self, axis: int, kp: float, ki: float, kd: float):
        """Set parameters for specific axis"""
        if 0 <= axis < self.num_axes:
            self.controllers[axis].set_parameters(kp, ki, kd)
    
    def set_axis_limits(self, axis: int, output_min: float, output_max: float):
        """Set output limits for specific axis"""
        if 0 <= axis < self.num_axes:
            self.controllers[axis].set_output_limits(output_min, output_max)
    
    def update(self, setpoints: List[float], process_variables: List[float],
               dt: Optional[float] = None) -> List[float]:
        """
        Update all axes
        
        Args:
            setpoints: List of setpoints for each axis
            process_variables: List of current values for each axis
            dt: Time step
            
        Returns:
            List of control outputs for each axis
        """
        if len(setpoints) != self.num_axes or len(process_variables) != self.num_axes:
            raise ValueError("Input dimensions must match number of axes")
        
        outputs = []
        for i in range(self.num_axes):
            output = self.controllers[i].update(setpoints[i], process_variables[i], dt)
            outputs.append(output)
        
        return outputs
    
    def reset_all(self):
        """Reset all controllers"""
        for controller in self.controllers:
            controller.reset()
    
    def set_axis_names(self, names: List[str]):
        """Set names for axes"""
        if len(names) == self.num_axes:
            self.axis_names = names


class AdaptivePIDController(PIDController):
    """
    Adaptive PID controller that adjusts parameters based on system response
    """
    
    def __init__(self, initial_kp: float = 1.0, initial_ki: float = 0.0, 
                 initial_kd: float = 0.0, adaptation_rate: float = 0.01):
        """
        Initialize adaptive PID controller
        
        Args:
            initial_kp: Initial proportional gain
            initial_ki: Initial integral gain
            initial_kd: Initial derivative gain
            adaptation_rate: Rate of parameter adaptation
        """
        super().__init__(initial_kp, initial_ki, initial_kd)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.adaptation_enabled = True
    
    def update(self, setpoint: float, process_variable: float, 
               dt: Optional[float] = None) -> float:
        """Update with adaptation"""
        output = super().update(setpoint, process_variable, dt)
        
        if self.adaptation_enabled and len(self.history['error']) > 10:
            self._adapt_parameters()
        
        return output
    
    def _adapt_parameters(self):
        """Adapt PID parameters based on performance"""
        # Simple adaptation based on error statistics
        recent_errors = self.history['error'][-10:]
        error_variance = np.var(recent_errors)
        mean_abs_error = np.mean(np.abs(recent_errors))
        
        # Store performance metric
        performance = error_variance + mean_abs_error
        self.performance_history.append(performance)
        
        # Adapt parameters if performance is poor
        if len(self.performance_history) > 20:
            recent_performance = np.mean(self.performance_history[-10:])
            old_performance = np.mean(self.performance_history[-20:-10])
            
            if recent_performance > old_performance:
                # Performance is getting worse, adjust parameters
                if mean_abs_error > 0.1:  # Large steady-state error
                    self.params.kp *= (1 + self.adaptation_rate)
                    self.params.ki *= (1 + self.adaptation_rate)
                elif error_variance > 0.05:  # High oscillation
                    self.params.kp *= (1 - self.adaptation_rate)
                    self.params.kd *= (1 + self.adaptation_rate)


class CascadePIDController:
    """
    Cascade PID controller with inner and outer loops
    """
    
    def __init__(self):
        """Initialize cascade PID controller"""
        self.outer_controller = PIDController()  # Position/outer loop
        self.inner_controller = PIDController()  # Velocity/inner loop
        
    def set_outer_parameters(self, kp: float, ki: float, kd: float):
        """Set outer loop parameters"""
        self.outer_controller.set_parameters(kp, ki, kd)
    
    def set_inner_parameters(self, kp: float, ki: float, kd: float):
        """Set inner loop parameters"""
        self.inner_controller.set_parameters(kp, ki, kd)
    
    def update(self, position_setpoint: float, current_position: float,
               current_velocity: float, dt: Optional[float] = None) -> float:
        """
        Update cascade controller
        
        Args:
            position_setpoint: Desired position
            current_position: Current position
            current_velocity: Current velocity
            dt: Time step
            
        Returns:
            Control output
        """
        # Outer loop: position control -> velocity setpoint
        velocity_setpoint = self.outer_controller.update(
            position_setpoint, current_position, dt
        )
        
        # Inner loop: velocity control -> control output
        control_output = self.inner_controller.update(
            velocity_setpoint, current_velocity, dt
        )
        
        return control_output
    
    def reset(self):
        """Reset both controllers"""
        self.outer_controller.reset()
        self.inner_controller.reset()


class PIDAutoTuner:
    """
    Automatic PID tuning using Ziegler-Nichols and other methods
    """
    
    def __init__(self):
        """Initialize PID auto-tuner"""
        self.tuning_data = {}
        
    def ziegler_nichols_tuning(self, ultimate_gain: float, 
                              ultimate_period: float,
                              controller_type: str = 'PID') -> Dict[str, float]:
        """
        Calculate PID parameters using Ziegler-Nichols method
        
        Args:
            ultimate_gain: Critical gain (Ku)
            ultimate_period: Critical period (Tu)
            controller_type: Type of controller ('P', 'PI', 'PID')
            
        Returns:
            Dictionary with suggested parameters
        """
        if controller_type == 'P':
            kp = 0.5 * ultimate_gain
            ki = 0.0
            kd = 0.0
        elif controller_type == 'PI':
            kp = 0.45 * ultimate_gain
            ki = 0.54 * ultimate_gain / ultimate_period
            kd = 0.0
        elif controller_type == 'PID':
            kp = 0.6 * ultimate_gain
            ki = 1.2 * ultimate_gain / ultimate_period
            kd = 0.075 * ultimate_gain * ultimate_period
        else:
            raise ValueError("Invalid controller type")
        
        return {'kp': kp, 'ki': ki, 'kd': kd}
    
    def cohen_coon_tuning(self, process_gain: float, time_constant: float,
                         dead_time: float) -> Dict[str, float]:
        """
        Calculate PID parameters using Cohen-Coon method
        
        Args:
            process_gain: Process gain
            time_constant: Process time constant
            dead_time: Process dead time
            
        Returns:
            Dictionary with suggested parameters
        """
        # Cohen-Coon formulas
        alpha = dead_time / time_constant
        
        kp = (1.35 / process_gain) * (time_constant / dead_time) * (1 + 0.18 * alpha)
        ki = kp / (time_constant * (2.5 - 2 * alpha) / (1 + 0.39 * alpha))
        kd = kp * time_constant * (0.37 - 0.37 * alpha) / (1 + 0.81 * alpha)
        
        return {'kp': kp, 'ki': ki, 'kd': kd}
    
    def lambda_tuning(self, process_gain: float, time_constant: float,
                     dead_time: float, desired_time_constant: float) -> Dict[str, float]:
        """
        Calculate PID parameters using Lambda tuning method
        
        Args:
            process_gain: Process gain
            time_constant: Process time constant
            dead_time: Process dead time
            desired_time_constant: Desired closed-loop time constant
            
        Returns:
            Dictionary with suggested parameters
        """
        lambda_c = desired_time_constant
        
        kp = time_constant / (process_gain * (lambda_c + dead_time))
        ki = kp / time_constant
        kd = 0.0  # Often not used in lambda tuning
        
        return {'kp': kp, 'ki': ki, 'kd': kd}