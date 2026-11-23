import numpy as np

from src.kalman_filter import KalmanFilter2D


def main():
    np.random.seed(42)

    # Actual cone position (unknown to the filter)
    true_position = np.array([10.0, 3.0])

    # Sensor measurement covariance
    R = np.eye(2) * 0.25

    # Generate repeated noisy measurements
    measurements = [
        true_position + np.random.normal(0, 0.5, 2)
        for _ in range(20)
    ]

    # Initialize using the first measurement
    kf = KalmanFilter2D(
        initial_position=measurements[0],
        initial_covariance=R.copy()
    )

    print("True position:", true_position)
    print("Initial estimate:", kf.x)

    for i, measurement in enumerate(measurements[1:], start=2):
        kf.predict()
        estimate = kf.update(measurement, R)

        error = np.linalg.norm(estimate - true_position)

        print(
            f"Measurement {i}: "
            f"Estimate = {np.round(estimate, 3)}, "
            f"Error = {error:.3f} m"
        )

    print("\nFinal estimated position:", kf.x)
    print("Final covariance:\n", kf.P)


if __name__ == "__main__":
    main()