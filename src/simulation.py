import numpy as np
import matplotlib.pyplot as plt
from src.detection import ConeDetection

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

def generate_curved_track(
    radius=20.0,
    track_width=6.0,
    num_cones=32
):
    angles = np.linspace(
        0, 2 * np.pi, num_cones,
        endpoint=False
    )

    inner_radius = radius - track_width / 2
    outer_radius = radius + track_width / 2

    blue_cones = np.column_stack((
        inner_radius * np.cos(angles),
        inner_radius * np.sin(angles)
    ))

    yellow_cones = np.column_stack((
        outer_radius * np.cos(angles),
        outer_radius * np.sin(angles)
    ))

    return blue_cones, yellow_cones


def simulate_curved_movement(
    radius=20.0,
    num_steps=180
):
    angles = np.linspace(
        0, 2 * np.pi, num_steps,
        endpoint=False
    )

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Vehicle heading is tangent to the circle
    heading = angles + np.pi / 2

    return np.column_stack((x, y, heading))

def generate_slalom_track(
    length=80.0,
    amplitude=5.0,
    wavelength=40.0,
    track_width=6.0,
    cone_spacing=4.0
):
    x = np.arange(0, length + 0.001, cone_spacing)

    k = 2 * np.pi / wavelength

    y = amplitude * np.sin(k * x)

    # Derivative of the centerline
    dy_dx = amplitude * k * np.cos(k * x)

    # Unit normal vectors perpendicular to the track
    normals = np.column_stack((
        -dy_dx,
        np.ones_like(x)
    ))

    normals /= np.linalg.norm(
        normals,
        axis=1,
        keepdims=True
    )

    centerline = np.column_stack((x, y))

    blue_cones = (
        centerline + normals * track_width / 2
    )

    yellow_cones = (
        centerline - normals * track_width / 2
    )

    return blue_cones, yellow_cones

def simulate_slalom_movement(
    length=80.0,
    amplitude=5.0,
    wavelength=40.0,
    step_size=1.0
):
    x = np.arange(0, length + 0.001, step_size)

    k = 2 * np.pi / wavelength

    y = amplitude * np.sin(k * x)

    dy_dx = amplitude * k * np.cos(k * x)

    headings = np.arctan2(
        dy_dx,
        np.ones_like(x)
    )

    return np.column_stack((x, y, headings))

def generate_track(track_type="straight"):

    if track_type == "straight":
        blue_cones, yellow_cones = generate_cones()
        vehicle_poses = simulate_vehicle_movement()

    elif track_type == "curved":
        blue_cones, yellow_cones = generate_curved_track()
        vehicle_poses = simulate_curved_movement()

    elif track_type == "slalom":
        blue_cones, yellow_cones = generate_slalom_track()
        vehicle_poses = simulate_slalom_movement()

    else:
        raise ValueError(
            f"Unknown track type: {track_type}"
        )

    return blue_cones, yellow_cones, vehicle_poses

def simulate_sensor_frame(
    cones,
    cone_classes,
    vehicle_pose,
    rng,
    timestamp=0.0,
    max_range=20.0,
    fov_degrees=120,
    noise_std=0.3
):
    x, y, heading = vehicle_pose

    relative_positions = cones - np.array([x, y])

    rotation = np.array([
        [np.cos(heading), np.sin(heading)],
        [-np.sin(heading), np.cos(heading)]
    ])

    local_positions = relative_positions @ rotation.T

    distances = np.linalg.norm(
        local_positions,
        axis=1
    )

    angles = np.arctan2(
        local_positions[:, 1],
        local_positions[:, 0]
    )

    # Determine geometric visibility
    visible = (
        (distances <= max_range)
        & (
            np.abs(angles)
            <= np.radians(fov_degrees / 2)
        )
    )

    visible_indices = np.flatnonzero(visible)

    detections = []

    covariance = np.eye(2) * noise_std ** 2

    for index in visible_indices:

        noise = rng.normal(
            0,
            noise_std,
            size=2
        )

        noisy_position = (
            local_positions[index] + noise
        )

        detections.append(
            ConeDetection(
                position=noisy_position,
                covariance=covariance.copy(),
                source="simulated",
                timestamp=timestamp,
                cone_class=cone_classes[index]
            )
        )

    return detections, visible_indices

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