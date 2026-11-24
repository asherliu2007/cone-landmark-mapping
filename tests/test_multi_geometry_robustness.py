import numpy as np
import matplotlib.pyplot as plt

from tests.test_multi_geometry import run_simulation


def main():

    track_types = [
        "straight",
        "curved",
        "slalom"
    ]

    noise_levels = [
        0.1,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0
    ]

    num_trials = 30

    all_results = {}

    total_experiments = (
        len(track_types)
        * len(noise_levels)
        * num_trials
    )

    completed = 0

    for track_type in track_types:

        all_results[track_type] = []

        print(f"\n{track_type.upper()} TRACK")
        print("=" * 40)

        for noise_std in noise_levels:

            results = []

            for seed in range(num_trials):

                metrics = run_simulation(
                    track_type=track_type,
                    seed=seed,
                    noise_std=noise_std,
                    verbose=False,
                    visualize=False
                )

                results.append(metrics)

                completed += 1

            precision = np.array([
                r["precision"] for r in results
            ])

            recall = np.array([
                r["recall"] for r in results
            ])

            mean_error = np.array([
                r["mean_error"] for r in results
            ])

            rmse = np.array([
                r["rmse"] for r in results
            ])

            false_positives = np.array([
                r["unmatched_landmarks"]
                for r in results
            ])

            missed_cones = np.array([
                r["missed_cones"]
                for r in results
            ])

            perfect_mapping = np.mean(
                np.isclose(precision, 1.0)
                & np.isclose(recall, 1.0)
            )

            summary = {
                "noise_std": noise_std,
                "precision": np.mean(precision),
                "recall": np.mean(recall),
                "mean_error": np.nanmean(mean_error),
                "rmse": np.nanmean(rmse),
                "false_positives": np.mean(false_positives),
                "missed_cones": np.mean(missed_cones),
                "perfect_mapping": perfect_mapping,
                "worst_precision": np.min(precision),
                "worst_recall": np.min(recall)
            }

            all_results[track_type].append(summary)

            print(f"\nNoise std: {noise_std:.2f} m")
            print("-------------------------")

            print(
                f"Precision: "
                f"{summary['precision']:.2%}"
            )

            print(
                f"Recall: "
                f"{summary['recall']:.2%}"
            )

            print(
                f"Mean error: "
                f"{summary['mean_error']:.4f} m"
            )

            print(
                f"RMSE: "
                f"{summary['rmse']:.4f} m"
            )

            print(
                f"False positives: "
                f"{summary['false_positives']:.2f}"
            )

            print(
                f"Missed cones: "
                f"{summary['missed_cones']:.2f}"
            )

            print(
                f"Perfect mapping: "
                f"{summary['perfect_mapping']:.2%}"
            )

            print(
                f"Worst precision: "
                f"{summary['worst_precision']:.2%}"
            )

            print(
                f"Worst recall: "
                f"{summary['worst_recall']:.2%}"
            )

            print(
                f"Progress: "
                f"{completed}/{total_experiments}"
            )

    plot_results(all_results)


def plot_results(all_results):

    plot_configs = [
        ("precision", "Precision", "Mapping Precision"),
        ("recall", "Recall", "Mapping Recall"),
        ("rmse", "RMSE (m)", "Mapping Position Error"),
        (
            "perfect_mapping",
            "Perfect Mapping Rate",
            "Perfect Mapping Reliability"
        ),
        (
            "false_positives",
            "Average False Positives",
            "False Landmark Creation"
        ),
        (
            "missed_cones",
            "Average Missed Cones",
            "Missed Landmark Detection"
        )
    ]

    for metric, ylabel, title in plot_configs:

        plt.figure(figsize=(10, 6))

        for track_type, results in all_results.items():

            noise_levels = [
                r["noise_std"] for r in results
            ]

            values = [
                r[metric] for r in results
            ]

            plt.plot(
                noise_levels,
                values,
                marker="o",
                label=track_type.capitalize()
            )

        if metric in ["precision", "recall"]:

            plt.axhline(
                y=0.95,
                color="gray",
                linestyle="--",
                label="Target"
            )

            plt.ylim(0, 1.05)

        elif metric == "rmse":

            plt.axhline(
                y=0.35,
                color="gray",
                linestyle="--",
                label="Target"
            )

        elif metric == "perfect_mapping":

            plt.ylim(0, 1.05)

        plt.xlabel("Measurement Noise Std. Dev. (m)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()