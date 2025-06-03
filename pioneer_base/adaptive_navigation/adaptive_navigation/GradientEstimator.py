import numpy as np

class GradientEstimator:
    def __init__(self):
        pass

    def estimate_gradient(self,
                 num_rovers: int, 
                 cluster_position,
                 rover_positions,
                 sensor_variable) -> tuple[float, float]:
        """
        Estimate the gradient of the sensor variable with respect to the cluster position.
        :param num_rovers: Number of rovers in the cluster.
        :param cluster_position: Current position of the cluster (x, y).
        :param rover_positions: List of rover positions (x, y, theta) for each rover.
        :param sensor_variable: List of sensor values corresponding to each rover.
        :return: Estimated gradient (dx, dy) of the sensor variable with respect to the cluster position.
        """
        if not isinstance(num_rovers, int) or num_rovers <= 0:
            raise ValueError("Number of rovers must be a positive integer.")
        if not isinstance(cluster_position, (list, tuple)) or len(cluster_position) < 2:
            raise ValueError("Cluster position must be a list or tuple with at least two elements (x, y).")
        if not len(sensor_variable) == num_rovers:
            raise ValueError("Number of sensor variables must match number of rovers.")
        if not int(len(rover_positions)//3) == num_rovers:
            raise ValueError("Number of rover positions must match number of rovers.")
        
        cluster_pos = cluster_position[0:2]
        position_matrix = [] # List of rover positions (x, y, bias)
        for i in range(num_rovers):
            position_matrix.append([rover_positions[i*3], 
                                    rover_positions[i*3+1], 
                                    1])
        position_matrix = np.array(position_matrix)
        sensor_variable = np.array(sensor_variable)
        # Calculate the gradient using least squares
        
        a, b, c = np.linalg.solve(
            position_matrix.T @ position_matrix,
            position_matrix.T @ sensor_variable
        )
        
        z_c = a * cluster_pos[0] + b * cluster_pos[1] + c

        return (a, b, z_c)
    
def main():
    pass
    


if __name__ == "__main__":
    main()