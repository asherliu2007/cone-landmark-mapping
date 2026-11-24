from dataclasses import dataclass

import numpy as np

from src.detection import ConeDetection


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


# Assumed physical cone heights in meters
CONE_HEIGHTS = {
    "blue_cone": 0.325,
    "yellow_cone": 0.325,
    "orange_cone": 0.325,
    "large_orange_cone": 0.505
}


def estimate_cone_position(detection, intrinsics):
    cone_class = detection["class"]

    if cone_class not in CONE_HEIGHTS:
        return None

    x1, y1, x2, y2 = detection["bbox"]

    pixel_height = abs(y2 - y1)
    center_x = (x1 + x2) / 2

    # Reject very small or invalid bounding boxes
    if pixel_height < 5:
        return None

    actual_height = CONE_HEIGHTS[cone_class]

    # Estimate forward depth
    depth = (
        intrinsics.fy * actual_height / pixel_height
    )

    # Estimate horizontal displacement
    lateral = (
        (center_x - intrinsics.cx)
        * depth / intrinsics.fx
    )

    # Vehicle coordinates:
    # x = forward, y = left
    position = np.array([
        depth,
        -lateral
    ])

    return position

def estimate_measurement_covariance(
    detection,
    intrinsics,
    pixel_height_std=2.0,
    pixel_center_std=2.0
):
    x1, y1, x2, y2 = detection["bbox"]

    h = abs(y2 - y1)
    u = (x1 + x2) / 2

    H = CONE_HEIGHTS[detection["class"]]

    Z = intrinsics.fy * H / h

    # Partial derivatives of forward position
    dx_dh = -Z / h
    dx_du = 0.0

    # Partial derivatives of lateral position
    dy_dh = (
        (u - intrinsics.cx)
        * Z / (intrinsics.fx * h)
    )

    dy_du = -Z / intrinsics.fx

    # Jacobian matrix
    J = np.array([
        [dx_dh, dx_du],
        [dy_dh, dy_du]
    ])

    # Bounding box measurement covariance
    pixel_covariance = np.diag([
        pixel_height_std ** 2,
        pixel_center_std ** 2
    ])

    # Propagate pixel uncertainty into meters
    covariance = J @ pixel_covariance @ J.T

    # Minimum uncertainty floor
    covariance += np.diag([
        0.1 ** 2,
        0.05 ** 2
    ])

    return covariance

def project_detections(
    detections,
    intrinsics,
    timestamp=0.0
):
    projected = []

    for detection in detections:

        position = estimate_cone_position(
            detection, intrinsics
        )

        if position is None:
            continue

        covariance = estimate_measurement_covariance(
            detection,
            intrinsics
        )

        projected.append(
            ConeDetection(
                position=position,
                covariance=covariance,
                source="camera",
                timestamp=timestamp,
                cone_class=detection["class"]
            )
        )

    return projected