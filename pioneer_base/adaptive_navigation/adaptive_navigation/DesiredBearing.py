import numpy as np

class GradientEstimator:
    def __init__(self, 
                 max_contour_gain=1.0, 
                 min_contour_gain=0.1,
                 max_correction_gain=2.0, 
                 min_correction_gain=0.05,
                 error_threshold=1.0,
                 gain_transition_rate=2.0):
        """
        Initialize the gradient estimator with adaptive gain control.
        :param max_contour_gain: Maximum gain for contour following (when near target)
        :param min_contour_gain: Minimum gain for contour following (when far from target)
        :param max_correction_gain: Maximum gain for error correction (when far from target)
        :param min_correction_gain: Minimum gain for error correction (when near target)
        :param error_threshold: Error value where gains transition
        :param gain_transition_rate: Rate of gain transition (higher = sharper transition)
        """
        self.max_contour_gain = max_contour_gain
        self.min_contour_gain = min_contour_gain
        self.max_correction_gain = max_correction_gain
        self.min_correction_gain = min_correction_gain
        self.error_threshold = error_threshold
        self.gain_transition_rate = gain_transition_rate

    def estimate(self,
                 nav_mode, # Navigation mode: "MAX", "MIN", "CONTUR"
                 gradient, # (x, y) tuple representing the gradient
                 scalar_value,  # Scalar value for bearing calculation
                 target_value=None # Optional target value for "CONTUR" mode
                 ) -> tuple[float, float]:
        """
        Estimate the desired bearing based on the navigation mode and gradient.
        :param nav_mode: Navigation mode, one of "MAX", "MIN", "CONTUR".
        :param gradient: Tuple (dx, dy) representing the gradient.
        :param scalar_value: Scalar value to be used in the bearing calculation.
        :param target_value: Target value for contour navigation.
        :return: Estimated desired bearing (dx, dy).
        """
        if not isinstance(nav_mode, str) or nav_mode not in ["MAX", "MIN", "CONTUR"]:
            raise ValueError("Navigation mode must be one of 'MAX', 'MIN', or 'CONTUR'.")
        if not isinstance(gradient, (list, tuple)) or len(gradient) != 2:
            raise ValueError("Gradient must be a tuple or list with two elements (dx, dy).")
        if not isinstance(scalar_value, (int, float)):
            raise ValueError("Scalar value must be an integer or float.")
        
        dx, dy = gradient
        
        if nav_mode == "MAX":
            # For maximum bearing, move in the direction of the gradient
            return dx, dy
        elif nav_mode == "MIN":
            # For minimum bearing, move opposite to the gradient
            return -dx, -dy
        elif nav_mode == "CONTUR":
            if target_value is None:
                raise ValueError("Target value must be provided for 'CONTUR' mode.")
            
            # Calculate gradient magnitude
            gradient_norm = np.sqrt(dx**2 + dy**2)
            if gradient_norm == 0:
                raise ValueError("Gradient cannot be zero for contour navigation.")
            
            # Calculate error between current scalar value and target value
            error = scalar_value - target_value
            error_magnitude = abs(error)
            
            # Adaptive gain calculation based on error magnitude
            contour_gain, correction_gain = self._calculate_adaptive_gains(error_magnitude)
            
            # Calculate clockwise perpendicular direction to gradient
            # For gradient vector (dx, dy), clockwise perpendicular is (dy, -dx)
            perp_dx = dy
            perp_dy = -dx
            
            # Normalize the perpendicular direction
            perp_norm = np.sqrt(perp_dx**2 + perp_dy**2)
            if perp_norm > 0:
                perp_dx_normalized = perp_dx / perp_norm
                perp_dy_normalized = perp_dy / perp_norm
            else:
                perp_dx_normalized = 0
                perp_dy_normalized = 0
            
            # Calculate error correction direction (perpendicular to contour)
            # If current value > target, move towards lower values (opposite to gradient)
            # If current value < target, move towards higher values (along gradient)
            if gradient_norm > 0:
                grad_dx_normalized = dx / gradient_norm
                grad_dy_normalized = dy / gradient_norm
            else:
                grad_dx_normalized = 0
                grad_dy_normalized = 0
            
            # Correction component (towards target value)
            correction_dx = -error * grad_dx_normalized * correction_gain
            correction_dy = -error * grad_dy_normalized * correction_gain
            
            # Contour following component (clockwise around contour)
            contour_dx = perp_dx_normalized * contour_gain
            contour_dy = perp_dy_normalized * contour_gain
            
            # Combine both components
            result_dx = contour_dx + correction_dx
            result_dy = contour_dy + correction_dy
            
            return result_dx, result_dy
        else:
            raise ValueError("Invalid navigation mode.")
    
    def _calculate_adaptive_gains(self, error_magnitude):
        """
        Calculate adaptive gains based on error magnitude.
        When far from target (large error): high correction gain, low contour gain
        When near target (small error): low correction gain, high contour gain
        :param error_magnitude: Absolute value of error from target
        :return: (contour_gain, correction_gain)
        """
        # Normalize error by threshold
        normalized_error = error_magnitude / self.error_threshold
        
        # Calculate decay factor using exponential function
        # When error is large: decay_factor approaches 0
        # When error is small: decay_factor approaches 1
        decay_factor = np.exp(-self.gain_transition_rate * normalized_error)
        
        # Contour gain: low when far (small decay_factor), high when near (large decay_factor)
        contour_gain = (self.min_contour_gain + 
                       (self.max_contour_gain - self.min_contour_gain) * decay_factor)
        
        # Correction gain: high when far (large normalized_error), low when near (small normalized_error)
        correction_factor = 1.0 - decay_factor  # Inverse of decay_factor
        correction_gain = (self.min_correction_gain + 
                          (self.max_correction_gain - self.min_correction_gain) * correction_factor)
        
        return contour_gain, correction_gain

    def set_adaptive_parameters(self, 
                               max_contour_gain=None, 
                               min_contour_gain=None,
                               max_correction_gain=None, 
                               min_correction_gain=None,
                               error_threshold=None,
                               gain_transition_rate=None):
        """
        Set the adaptive gain parameters.
        :param max_contour_gain: Maximum gain for contour following (when near target)
        :param min_contour_gain: Minimum gain for contour following (when far from target)
        :param max_correction_gain: Maximum gain for error correction (when far from target)
        :param min_correction_gain: Minimum gain for error correction (when near target)
        :param error_threshold: Error value where gains transition
        :param gain_transition_rate: Rate of gain transition
        """
        if max_contour_gain is not None:
            self.max_contour_gain = max_contour_gain
        if min_contour_gain is not None:
            self.min_contour_gain = min_contour_gain
        if max_correction_gain is not None:
            self.max_correction_gain = max_correction_gain
        if min_correction_gain is not None:
            self.min_correction_gain = min_correction_gain
        if error_threshold is not None:
            self.error_threshold = error_threshold
        if gain_transition_rate is not None:
            self.gain_transition_rate = gain_transition_rate
    
    def get_current_gains(self, error_magnitude):
        """
        Get the current gains for a given error magnitude.
        Useful for debugging and visualization.
        :param error_magnitude: Absolute value of error from target
        :return: (contour_gain, correction_gain)
        """
        return self._calculate_adaptive_gains(error_magnitude)

def main():
    # Example usage with adaptive gain control
    estimator = GradientEstimator(
        max_contour_gain=1.5, 
        min_contour_gain=0.1,
        max_correction_gain=3.0, 
        min_correction_gain=0.05,
        error_threshold=2.0,
        gain_transition_rate=1.5
    )
    
    # Test cases for basic functionality
    test_cases = [
        # (nav_mode, gradient, scalar_value, target_value, description)
        ("MAX", (1.0, 0.5), 10.0, None, "Move towards maximum"),
        ("MIN", (1.0, 0.5), 10.0, None, "Move towards minimum"),
        ("CONTUR", (1.0, 0.0), 12.0, 10.0, "Contour following - gradient pointing east, far from target"),
        ("CONTUR", (0.0, 1.0), 8.0, 10.0, "Contour following - gradient pointing north, far from target"),
        ("CONTUR", (1.0, 0.0), 10.2, 10.0, "Contour following - gradient pointing east, near target"),
        ("CONTUR", (0.7071, 0.7071), 9.8, 10.0, "Contour following - gradient pointing northeast, near target"),
    ]
    
    print("Gradient Estimator Test Results:")
    print("-" * 70)
    
    for nav_mode, gradient, scalar_value, target_value, description in test_cases:
        try:
            result = estimator.estimate(nav_mode, gradient, scalar_value, target_value)
            print(f"{description}:")
            print(f"  Input: mode={nav_mode}, gradient={gradient}, value={scalar_value}, target={target_value}")
            if nav_mode == "CONTUR":
                error_mag = abs(scalar_value - target_value)
                contour_gain, correction_gain = estimator.get_current_gains(error_mag)
                print(f"  Error: {error_mag:.3f}, Gains: contour={contour_gain:.3f}, correction={correction_gain:.3f}")
            print(f"  Output: bearing=({result[0]:.3f}, {result[1]:.3f})")
            print()
        except Exception as e:
            print(f"Error in {description}: {e}")
            print()
    
    # Demonstrate adaptive gain behavior
    print("Adaptive Gain Demonstration:")
    print("-" * 70)
    print("Error Magnitude | Contour Gain | Correction Gain | Behavior")
    print("-" * 70)
    
    test_errors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for error in test_errors:
        contour_gain, correction_gain = estimator.get_current_gains(error)
        if error <= estimator.error_threshold * 0.5:
            behavior = "Contour following dominant"
        elif error <= estimator.error_threshold * 2.0:
            behavior = "Balanced"
        else:
            behavior = "Error correction dominant"
        print(f"{error:13.1f} | {contour_gain:11.3f} | {correction_gain:14.3f} | {behavior}")
    
    print("\nNote: Gains automatically adjust based on distance from target value.")
    print("Far from target: High correction gain, low contour gain")
    print("Near target: Low correction gain, high contour gain")

if __name__ == "__main__":
    main()