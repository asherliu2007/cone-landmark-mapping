import numpy as np
from src.detection import ConeDetection


def main():
    detection = ConeDetection(
        position=[8.5, 2.1],
        covariance=[
            [0.09, 0.0],
            [0.0, 0.09]
        ],
        source="camera",
        timestamp=0.1,
        cone_class="blue_cone"
    )

    print("Position:", detection.position)
    print("Covariance:\n", detection.covariance)
    print("Source:", detection.source)
    print("Timestamp:", detection.timestamp)
    print("Class:", detection.cone_class)


if __name__ == "__main__":
    main()