"""
Script to visualize Python vs Swift genetic programming benchmarks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_gp_benchmarks():
    """Generate comparison graphs for Python vs Swift genetic programming benchmarks"""
    # Set visualization style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["savefig.dpi"] = 300

    # Color palette for language comparison
    COLORS = {"Python": "#306998", "Swift": "#F05138"}

    # Load benchmark data
    python_data = pd.read_csv("../../data/benchmarking/python_gp_benchmarks.csv")
    swift_data = pd.read_csv("../../data/benchmarking/swift_gp_benchmarks.csv")

    # Add language identifiers and combine datasets
    python_data["language"] = "Python"
    swift_data["language"] = "Swift"
    combined_data = pd.concat([python_data, swift_data], ignore_index=True)

    # Filter data for each parameter type
    pop_data = combined_data[combined_data["parameter"] == "Population Size"]
    gen_data = combined_data[combined_data["parameter"] == "Generations"]

    # Runtime comparison graph
    plt.figure(figsize=(16, 8))

    # Runtime vs Population Size
    plt.subplot(1, 2, 1)
    for language, color in COLORS.items():
        data = pop_data[pop_data["language"] == language]
        plt.plot(
            data["value"],
            data["time"],
            "o-",
            color=color,
            linewidth=2.5,
            markersize=10,
            label=language,
        )
    plt.xlabel("Population Size")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Population Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Runtime vs Generations
    plt.subplot(1, 2, 2)
    for language, color in COLORS.items():
        data = gen_data[gen_data["language"] == language]
        plt.plot(
            data["value"],
            data["time"],
            "o-",
            color=color,
            linewidth=2.5,
            markersize=10,
            label=language,
        )
    plt.xlabel("Number of Generations")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Generations")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle(
        "Genetic Programming Runtime Comparison: Python vs Swift",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("../../img/benchmarking/gp_runtime_comparison.png")

    # Accuracy comparison graph
    plt.figure(figsize=(16, 8))

    # Accuracy vs Population Size
    plt.subplot(1, 2, 1)
    for language, color in COLORS.items():
        data = pop_data[pop_data["language"] == language]
        plt.plot(
            data["value"],
            data["accuracy"],
            "o-",
            color=color,
            linewidth=2.5,
            markersize=10,
            label=language,
        )
    plt.xlabel("Population Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Population Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy vs Generations
    plt.subplot(1, 2, 2)
    for language, color in COLORS.items():
        data = gen_data[gen_data["language"] == language]
        plt.plot(
            data["value"],
            data["accuracy"],
            "o-",
            color=color,
            linewidth=2.5,
            markersize=10,
            label=language,
        )
    plt.xlabel("Number of Generations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Generations")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle(
        "Genetic Programming Accuracy Comparison: Python vs Swift",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("../../img/benchmarking/gp_accuracy_comparison.png")

    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Python average runtime: {python_data['time'].mean():.2f} seconds")
    print(f"Swift average runtime: {swift_data['time'].mean():.2f} seconds")
    print(f"Python average accuracy: {python_data['accuracy'].mean():.3f}")
    print(f"Swift average accuracy: {swift_data['accuracy'].mean():.3f}")

    print("\nVisualization files created:")
    print("- ../../img/benchmarking/gp_runtime_comparison.png")
    print("- ../../img/benchmarking/gp_accuracy_comparison.png")


if __name__ == "__main__":
    compare_gp_benchmarks()
