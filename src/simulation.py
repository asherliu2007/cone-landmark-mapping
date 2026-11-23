import numpy as np
import matplotlib.pyplot as plt


def generate_cones():
    # X positions along the track
    x = np.arange(0, 50, 5)

    # Blue cones on the left
    blue_cones = np.column_stack((x, np.full(len(x), 3)))

    # Yellow cones on the right
    yellow_cones = np.column_stack((x, np.full(len(x), -3)))

    return blue_cones, yellow_cones

def add_noise(cones, noise_std=0.5):
    noise = np.random.normal(0, noise_std, cones.shape)
    return cones + noise


def plot_cones(blue_cones, yellow_cones):
    plt.figure(figsize=(10, 5))

    plt.scatter(
        blue_cones[:, 0],
        blue_cones[:, 1],
        color="blue",
        label="Blue Cones"
    )

    plt.scatter(
        yellow_cones[:, 0],
        yellow_cones[:, 1],
        color="gold",
        label="Yellow Cones"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("FSAE Driverless Cone Map")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()

def simulate_vehicle_movement():
    # Vehicle travels along the center of the track
    x_positions = np.arange(0, 50, 2)

    # Vehicle stays at y = 0 and faces forward
    y_positions = np.zeros(len(x_positions))
    headings = np.zeros(len(x_positions))

    vehicle_poses = np.column_stack(
        (x_positions, y_positions, headings)
    )

    return vehicle_poses

def simulate_cone_detections(
    cones,
    vehicle_pose,
    max_range=20.0,
    fov_degrees=120,
    noise_std=0.3
):
    x, y, heading = vehicle_pose

    # Convert global cone positions to vehicle coordinates
    relative_positions = cones - np.array([x, y])

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    rotation = np.array([
        [cos_h, sin_h],
        [-sin_h, cos_h]
    ])

    local_positions = relative_positions @ rotation.T

    # Calculate distance and angle to each cone
    distances = np.linalg.norm(local_positions, axis=1)

    angles = np.arctan2(
        local_positions[:, 1],
        local_positions[:, 0]
    )

    # Keep only cones within sensor range and FOV
    visible = (
        (distances <= max_range) &
        (np.abs(angles) <= np.radians(fov_degrees / 2))
    )

    detected_cones = local_positions[visible]

    # Add Gaussian measurement noise
    noise = np.random.normal(
        0,
        noise_std,
        detected_cones.shape
    )

    return detected_cones + noise

def plot_sensor_detections(
    blue_cones,
    yellow_cones,
    vehicle_pose,
    detections
):
    plt.figure(figsize=(12, 6))

    plt.scatter(
        blue_cones[:, 0],
        blue_cones[:, 1],
        color="blue",
        label="Blue Cones"
    )

    plt.scatter(
        yellow_cones[:, 0],
        yellow_cones[:, 1],
        color="gold",
        label="Yellow Cones"
    )

    # Convert noisy detections back to global coordinates
    x, y, heading = vehicle_pose

    rotation = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])

    global_detections = (
        detections @ rotation.T + np.array([x, y])
    )

    plt.scatter(
        global_detections[:, 0],
        global_detections[:, 1],
        color="red",
        marker="x",
        s=80,
        label="Sensor Detections"
    )

    plt.scatter(
        x,
        y,
        color="green",
        s=100,
        label="Vehicle"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Simulated Cone Sensor Detections")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_vehicle_movement(blue_cones, yellow_cones, vehicle_poses):
    plt.figure(figsize=(12, 6))

    plt.scatter(
        blue_cones[:, 0],
        blue_cones[:, 1],
        color="blue",
        label="Blue Cones"
    )

    plt.scatter(
        yellow_cones[:, 0],
        yellow_cones[:, 1],
        color="gold",
        label="Yellow Cones"
    )

    plt.plot(
        vehicle_poses[:, 0],
        vehicle_poses[:, 1],
        color="green",
        linewidth=2,
        label="Vehicle Path"
    )

    plt.scatter(
        vehicle_poses[0, 0],
        vehicle_poses[0, 1],
        color="green",
        marker="o",
        s=100,
        label="Start"
    )

    plt.scatter(
        vehicle_poses[-1, 0],
        vehicle_poses[-1, 1],
        color="red",
        marker="x",
        s=100,
        label="End"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Simulated Vehicle Movement")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_noisy_cones(actual_cones, noisy_cones):
    plt.figure(figsize=(10, 5))

    plt.scatter(
        actual_cones[:, 0],
        actual_cones[:, 1],
        color="blue",
        label="Actual Cone Positions"
    )

    plt.scatter(
        noisy_cones[:, 0],
        noisy_cones[:, 1],
        color="red",
        marker="x",
        label="Noisy Measurements"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Actual vs Noisy Cone Positions")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()

if __name__ == "__main__":
    blue_cones, yellow_cones = generate_cones()

    actual_cones = np.vstack((blue_cones, yellow_cones))

    vehicle_poses = simulate_vehicle_movement()

    # Select one vehicle position for testing
    vehicle_pose = vehicle_poses[5]

    detections = simulate_cone_detections(
        actual_cones,
        vehicle_pose
    )

    print(f"Vehicle position: {vehicle_pose}")
    print(f"Detected {len(detections)} cones")

    plot_sensor_detections(
        blue_cones,
        yellow_cones,
        vehicle_pose,
        detections
    )