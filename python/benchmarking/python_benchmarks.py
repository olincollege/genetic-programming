"""
Benchmarking module for genetic programming performance comparison.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing modules with the appropriate paths
from parse_tree import ParseTree, TerminalGenerationRules, FunctionNode, TerminalNode
from genetic_programming import IrisGP


def run_benchmarks():
    """Run performance benchmarks for Python GP implementation"""
    print("Running Python GP benchmarks...\n")

    # Load iris dataset
    iris = pd.read_csv("../../data/Iris.csv")
    train_df, test_df = train_test_split(iris, test_size=0.3, random_state=3)

    # Configuration for benchmarks
    population_sizes = [50, 100, 200]
    generation_counts = [10, 20, 50]

    results = []

    # Default values for GP
    FUNCTION_SET = ["+", "-", "*", "/"]
    TERMINAL_RULES = TerminalGenerationRules(
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        (-10, 10),
        ints_only=False,
        no_random_constants=False,
    )
    MAX_DEPTH = 4
    TERMINAL_PROBABILITY = 0.2
    gp = IrisGP(
        FUNCTION_SET,
        TERMINAL_RULES,
        MAX_DEPTH,
        TERMINAL_PROBABILITY,
    )

    POPULATION_SIZE = 100
    GENERATIONS = 20
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    CHAMPION_SURVIVAL_PERCENTAGE = 0.1

    # Run benchmarks with varying population size (smaller for faster testing)
    print("\nBenchmarking with varying population size...")
    for pop_size in population_sizes:
        start_time = time.time()
        best, _, _ = gp.solve(
            pop_size,
            GENERATIONS,
            CROSSOVER_RATE,
            MUTATION_RATE,
            CHAMPION_SURVIVAL_PERCENTAGE,
            train_df,
        )
        end_time = time.time()
        total_time = end_time - start_time
        accuracy = IrisGP.evaluate_fitness(best, test_df) / len(test_df)
        results.append(
            {
                "parameter": "Population Size",
                "value": pop_size,
                "time": total_time,
                "accuracy": accuracy,
            }
        )

        print(
            f"Population size {pop_size}: {total_time:.2f} seconds, Accuracy: {accuracy:.4f}"
        )

    # Run benchmarks with varying generations (smaller for faster testing)
    print("\nBenchmarking with varying generation count...")
    for gen_count in generation_counts:
        start_time = time.time()
        best, _, _ = gp.solve(
            POPULATION_SIZE,
            gen_count,
            CROSSOVER_RATE,
            MUTATION_RATE,
            CHAMPION_SURVIVAL_PERCENTAGE,
            train_df,
        )
        end_time = time.time()
        total_time = end_time - start_time
        accuracy = IrisGP.evaluate_fitness(best, test_df) / len(test_df)

        results.append(
            {
                "parameter": "Generations",
                "value": gen_count,
                "time": total_time,
                "accuracy": accuracy,
            }
        )

        print(
            f"Generation count {gen_count}: {total_time:.2f} seconds, Accuracy: {accuracy:.4f}"
        )

    # Save results to CSV
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            "../../data/benchmarking/python_gp_benchmarks.csv", index=False
        )

        # Ensure output directory exists
        os.makedirs("../../img", exist_ok=True)

        # Get filtered data
        pop_results = results_df[results_df["parameter"] == "Population Size"]
        gen_results = results_df[results_df["parameter"] == "Generations"]

        # Plot runtime results
        plt.figure(figsize=(12, 6))

        # Plot population size benchmarks
        plt.subplot(1, 2, 1)
        plt.plot(
            pop_results["value"],
            pop_results["time"],
            "o-",
            color="#306998",
            label="Execution Time",
        )
        plt.xlabel("Population Size")
        plt.ylabel("Time (seconds)")
        plt.title("Execution Time vs Population Size")
        plt.grid(True)

        # Plot generation benchmarks
        plt.subplot(1, 2, 2)
        plt.plot(
            gen_results["value"],
            gen_results["time"],
            "o-",
            color="#306998",
            label="Execution Time",
        )
        plt.xlabel("Generations")
        plt.ylabel("Time (seconds)")
        plt.title("Execution Time vs Generations")
        plt.grid(True)

        plt.suptitle("Python Genetic Programming: Runtime Performance", fontsize=14)
        plt.tight_layout()
        plt.savefig("../../img/python_benchmarks.png")
        plt.close()

        # Plot accuracy results
        plt.figure(figsize=(12, 6))

        # Plot population size vs accuracy
        plt.subplot(1, 2, 1)
        plt.plot(
            pop_results["value"],
            pop_results["accuracy"],
            "o-",
            color="#306998",
            label="Accuracy",
        )
        plt.xlabel("Population Size")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Population Size")
        plt.grid(True)

        # Plot generations vs accuracy
        plt.subplot(1, 2, 2)
        plt.plot(
            gen_results["value"],
            gen_results["accuracy"],
            "o-",
            color="#306998",
            label="Accuracy",
        )
        plt.xlabel("Generations")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Generations")
        plt.grid(True)

        plt.suptitle("Python Genetic Programming: Classification Accuracy", fontsize=14)
        plt.tight_layout()
        plt.savefig("../../img/python_accuracy.png")
        plt.close()

    except Exception as e:
        print(f"Error creating plots: {str(e)}")

    return results


if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    if benchmark_results is not None:
        print(
            "\nBenchmark results saved to '../../data/benchmarking/python_gp_benchmarks.csv'"
        )
        print("Benchmark plots saved to:")
        print("- '../../img/benchmarking/python_benchmarks.png' (Runtime)")
        print("- '../../img/benchmarking/python_accuracy.png' (Accuracy)")
