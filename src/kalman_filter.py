import numpy as np


class KalmanFilter2D:
    def __init__(self, initial_position, initial_covariance):
        # Estimated cone position [x, y]
        self.x = np.asarray(initial_position, dtype=float)

        # Uncertainty of the estimate
        self.P = np.asarray(initial_covariance, dtype=float)

        # Small process noise
        self.Q = np.eye(2) * 0.001

    def predict(self):
        # Stationary landmark: position does not change
        self.P = self.P + self.Q

        return self.x

    def update(self, measurement, measurement_covariance):
        z = np.asarray(measurement, dtype=float)
        R = np.asarray(measurement_covariance, dtype=float)

        # Measurement residual
        y = z - self.x

        # Innovation covariance
        S = self.P + R

        # Kalman gain
        K = np.linalg.solve(S.T, self.P.T).T

        # Update estimated position
        self.x = self.x + K @ y

        # Update covariance (Joseph form)
        I = np.eye(2)
        self.P = (I - K) @ self.P @ (I - K).T + K @ R @ K.T

        return self.x