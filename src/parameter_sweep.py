import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

from genetic_programming import IrisGP
from parse_tree import TerminalGenerationRules


class ParameterSweep:
    """
    Class containing methods for parameter sweeping of the GP model.

    Attributes:
        DEFAULT_PARAMS (dict): Default values for each parameter of the GP model.
    """

    DEFAULT_PARAMS = {
        "function_set": ["+", "-", "*", "/"],
        "terminal_rules": TerminalGenerationRules(
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            (-10, 10),
            ints_only=False,
            no_random_constants=False,
        ),
        "max_depth": 3,
        "terminal_probability": 0.2,
        "population_size": 50,
        "generations": 20,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "champion_survival_percentage": 0.1,
    }

    @staticmethod
    def sweep_all_parameters(param_grid, train_df, test_df, iterations=10):
        """
        Sweep through the parameter grid and evaluate the GP model.
        Args:
            param_grid: Dictionary of parameters to sweep. Keys include
                population_size, generations, crossover_rate, mutation_rate,
                champion_survival_percentage, and max_depth. Values are the
                lists of the parameters.
            train_df: Training DataFrame.
            test_df: Testing DataFrame.
            iterations: Number of iterations for averaging.
        Returns:
            DataFrame with results.
        """
        out = pd.DataFrame()
        for param_name, values in param_grid.items():
            results = ParameterSweep.sweep_single_parameter(
                param_name,
                values,
                train_df,
                test_df,
                iterations=iterations,
            )
            out = pd.concat([out, results])
        return out

    @staticmethod
    def sweep_single_parameter(
        param_name, param_values, train_df, test_df, iterations=10
    ):
        """
        Sweep through a single parameter and evaluate the GP model.
        Args:
            param_name: Name of the parameter to sweep.
            param_values: List of values for the parameter.
            train_df: Training DataFrame.
            test_df: Testing DataFrame.
            iterations: Number of iterations for averaging.
        Returns:
            DataFrame with results.
        """
        # Initialize results list
        results = []

        print(f"Sweeping {param_name}...")
        for val in param_values:
            print(f"{param_name} = {val}")
            params = ParameterSweep.DEFAULT_PARAMS.copy()
            params[param_name] = val

            accuracies = []
            for _ in range(iterations):  # multiple trials for averaging
                gp = IrisGP(
                    params["function_set"],
                    params["terminal_rules"],
                    params["max_depth"],
                    params["terminal_probability"],
                )
                best_tree, _, _ = gp.solve(
                    population_size=params["population_size"],
                    generations=params["generations"],
                    crossover_rate=params["crossover_rate"],
                    mutation_rate=params["mutation_rate"],
                    champion_survival_percentage=params["champion_survival_percentage"],
                    train_df=train_df,
                )
                acc = IrisGP.evaluate_fitness(best_tree, test_df) / len(test_df)
                accuracies.append(acc)
                # print(f"Accuracy: {acc}")

            avg_acc = sum(accuracies) / len(accuracies)
            results.append({"param": param_name, "value": val, "accuracy": avg_acc})

        # Convert to DataFrame for plotting
        df = pd.DataFrame(results)
        return df

    @staticmethod
    def plot_paramter_sweep_results(param_grid: dict, df: pd.DataFrame):
        """
        Plot the results of the parameter sweep.
        Args:
            param_grid: Dictionary of parameters to sweep. Keys include
                population_size, generations, crossover_rate, mutation_rate,
                champion_survival_percentage, and max_depth. Values are the
                lists of the parameters.
            df: DataFrame with the parameter sweep results.
        """
        sns.set_theme(style="whitegrid")
        for param in param_grid.keys():
            subset = df[df["param"] == param]
            plt.figure(figsize=(6, 4))
            sns.lineplot(x="value", y="accuracy", data=subset, marker="o")
            plt.title(f"Effect of {param} on Accuracy")
            plt.xlabel(param)
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.ylim(0.75, 1)

            if param == "mutation_rate":
                plt.xscale("log")

            plt.show()

    @staticmethod
    def load_from_csv(results_path: str, param_grid: dict) -> pd.DataFrame:
        """
        Load parameter sweep results from CSV files and concatenate into a
        single DataFrame.

        Args:
            results_path: Path to the directory containing the CSV files.
            param_grid: Dictionary of parameters to sweep. Keys include
                population_size, generations, crossover_rate, mutation_rate,
                champion_survival_percentage, and max_depth. Values are the
                lists of the parameters.
        """
        out = pd.DataFrame()
        for param_name in param_grid.keys():
            df = pd.read_csv(f"{results_path}/param_sweep_{param_name}.csv")
            out = pd.concat([out, df])
        return out
