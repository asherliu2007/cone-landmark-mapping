import time
import numpy as np
import platform

from src.landmark_mapping import LandmarkMapper

from src.simulation import (
    generate_track,
    simulate_sensor_frame
)


def benchmark_mapping(
    track_type,
    num_trials=10,
    noise_std=0.3
):
    latencies = []
    detection_counts = []
    landmark_counts = []

    blue_cones, yellow_cones, vehicle_poses = (
        generate_track(track_type)
    )

    actual_cones = np.vstack((
        blue_cones,
        yellow_cones
    ))

    actual_classes = np.array(
        ["blue_cone"] * len(blue_cones)
        + ["yellow_cone"] * len(yellow_cones)
    )

    # Trial -1 is an unrecorded warm-up trial
    for seed in range(-1, num_trials):

        rng = np.random.default_rng(seed + 1)

        mapper = LandmarkMapper(
            association_threshold=5.991,
            min_observations=2
        )

        for step, pose in enumerate(vehicle_poses):

            # Generate detections outside the timed section
            detections, _ = simulate_sensor_frame(
                cones=actual_cones,
                cone_classes=actual_classes,
                vehicle_pose=pose,
                rng=rng,
                timestamp=float(step),
                max_range=20.0,
                fov_degrees=120,
                noise_std=noise_std
            )

            # Measure mapping processing time
            start = time.perf_counter_ns()

            mapper.update(detections, pose)

            end = time.perf_counter_ns()

            latency_ms = (end - start) / 1_000_000

            # Exclude warm-up measurements
            if seed >= 0:

                latencies.append(latency_ms)

                detection_counts.append(
                    len(detections)
                )

                landmark_counts.append(
                    len(mapper.landmarks)
                )

    latencies = np.array(latencies)

    mean_latency = np.mean(latencies)

    return {
        "frames_processed": len(latencies),
        "mean_latency": mean_latency,
        "median_latency": np.median(latencies),
        "std_latency": np.std(latencies),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "max_latency": np.max(latencies),
        "throughput": 1000 / mean_latency,
        "mean_detections": np.mean(detection_counts),
        "max_detections": np.max(detection_counts),
        "mean_landmarks": np.mean(landmark_counts),
        "max_landmarks": np.max(landmark_counts),
        "frames_over_10ms": np.mean(latencies > 10),
        "frames_over_20ms": np.mean(latencies > 20)
    }


def main():

    print("Mapping Performance Benchmark")
    print("=" * 40)

    print(f"System: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")

    track_types = [
        "straight",
        "curved",
        "slalom"
    ]

    for track_type in track_types:

        results = benchmark_mapping(
            track_type=track_type,
            num_trials=10,
            noise_std=0.3
        )

        print(f"\n{track_type.upper()} TRACK")
        print("=" * 40)

        print(
            f"Frames processed: "
            f"{results['frames_processed']}"
        )

        print(
            f"Mean latency: "
            f"{results['mean_latency']:.4f} ms"
        )

        print(
            f"Median latency: "
            f"{results['median_latency']:.4f} ms"
        )

        print(
            f"Latency std: "
            f"{results['std_latency']:.4f} ms"
        )

        print(
            f"P95 latency: "
            f"{results['p95_latency']:.4f} ms"
        )

        print(
            f"P99 latency: "
            f"{results['p99_latency']:.4f} ms"
        )

        print(
            f"Maximum latency: "
            f"{results['max_latency']:.4f} ms"
        )

        print(
            f"Estimated throughput: "
            f"{results['throughput']:.2f} FPS"
        )

        print(
            f"Average detections: "
            f"{results['mean_detections']:.2f}"
        )

        print(
            f"Maximum detections: "
            f"{results['max_detections']}"
        )

        print(
            f"Average landmarks: "
            f"{results['mean_landmarks']:.2f}"
        )

        print(
            f"Maximum landmarks: "
            f"{results['max_landmarks']}"
        )

        print(
            f"Frames exceeding 10 ms: "
            f"{results['frames_over_10ms']:.2%}"
        )

        print(
            f"Frames exceeding 20 ms: "
            f"{results['frames_over_20ms']:.2%}"
        )


if __name__ == "__main__":
    main()