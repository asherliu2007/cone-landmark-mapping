import numpy as np

from tests.test_fsoco_multiframe import main as run_simulation


def main():

    num_trials = 100
    results = []

    for seed in range(num_trials):

        metrics = run_simulation(
            seed=seed,
            verbose=False,
            visualize=False
        )

        results.append(metrics)

        if (seed + 1) % 10 == 0:
            print(
                f"Completed {seed + 1}/{num_trials} trials"
            )

    print("\nFSOCO Robustness Evaluation")
    print("---------------------------")
    print(f"Total trials: {num_trials}")

    metric_names = [
        "precision",
        "recall",
        "mean_error",
        "rmse",
        "max_error",
        "matched_landmarks",
        "unmatched_landmarks",
        "missed_cones"
    ]

    for metric in metric_names:

        values = np.array([
            result[metric]
            for result in results
        ])

        valid_values = values[np.isfinite(values)]

        print(f"\n{metric}:")

        if len(valid_values) == 0:
            print("No valid measurements")
            continue

        print(f"  Mean: {np.mean(valid_values):.4f}")
        print(f"  Std:  {np.std(valid_values):.4f}")
        print(f"  Min:  {np.min(valid_values):.4f}")
        print(f"  Max:  {np.max(valid_values):.4f}")
        print(f"  Valid trials: {len(valid_values)}")

    perfect_precision = sum(
        np.isclose(result["precision"], 1.0)
        for result in results
    )

    perfect_recall = sum(
        np.isclose(result["recall"], 1.0)
        for result in results
    )

    perfect_mapping = sum(
        np.isclose(result["precision"], 1.0)
        and np.isclose(result["recall"], 1.0)
        for result in results
    )

    print("\nPerfect Mapping Statistics")
    print("--------------------------")

    print(
        f"Perfect precision: "
        f"{100 * perfect_precision / num_trials:.1f}%"
    )

    print(
        f"Perfect recall: "
        f"{100 * perfect_recall / num_trials:.1f}%"
    )

    print(
        f"Perfect precision and recall: "
        f"{100 * perfect_mapping / num_trials:.1f}%"
    )

    worst_precision_seed = min(
        range(num_trials),
        key=lambda i: results[i]["precision"]
    )

    worst_recall_seed = min(
        range(num_trials),
        key=lambda i: results[i]["recall"]
    )

    print("\nWorst-case Seeds")
    print("----------------")
    print(f"Worst precision seed: {worst_precision_seed}")
    print(f"Worst recall seed: {worst_recall_seed}")


if __name__ == "__main__":
    main()