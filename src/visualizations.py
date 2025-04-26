import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from parse_tree import *
from genetic_programming import IrisGP


class Visualizations:
    def __init__(self, parse_tree: ParseTree, test_df: pd.DataFrame):
        self.parse_tree: ParseTree = parse_tree
        self.test_df: pd.DataFrame = test_df
        self.test_results: pd.DataFrame = self.tabulate_results()

    def tabulate_results(self):
        test_results = self.test_df.copy()
        test_results["Correct"] = None
        test_results["PredictedValue"] = None
        test_results["PredictedSpecies"] = None
        for idx, row in test_results.iterrows():
            correctness, res = IrisGP.evaluate_row(self.parse_tree, row)
            test_results.at[idx, "Correct"] = correctness
            test_results.at[idx, "PredictedValue"] = res
            if res < IrisGP.THRESHOLD_LOW:
                species = "Iris-setosa"
            elif res < IrisGP.THRESHOLD_HIGH:
                species = "Iris-versicolor"
            else:
                species = "Iris-virginica"
            test_results.at[idx, "PredictedSpecies"] = species
        return test_results

    def plot_predictions_by_dimension(self, part: str):
        COLOR_MAP = {
            "Iris-setosa": "tab:blue",
            "Iris-versicolor": "tab:orange",
            "Iris-virginica": "tab:green",
        }
        CORRECT_OPACITY = 0.3
        CORRECT_SIZE = 30
        INCORRECT_SIZE = 60
        CORRECT_LEGEND_SIZE = 6
        INCORRECT_LEGEND_SIZE = 10

        def legend_point(size: int, alpha: int, label: str):
            return Line2D(
                [],
                [],
                color="gray",
                marker="o",
                linestyle="None",
                markersize=size,
                alpha=alpha,
                label=label,
            )

        for _, row in self.test_results.iterrows():
            plt.scatter(
                row[f"{part}LengthCm"],
                row[f"{part}WidthCm"],
                color=COLOR_MAP[row["PredictedSpecies"]],
                # edgecolors=COLOR_MAP[row["Species"]],
                # linewidths=2,
                alpha=CORRECT_OPACITY if row["Correct"] else 1,
                s=CORRECT_SIZE if row["Correct"] else INCORRECT_SIZE,
            )

        handles = [
            Patch(color="tab:blue", label="Predicted: Setosa"),
            Patch(color="tab:orange", label="Predicted: Versicolor"),
            Patch(color="tab:green", label="Predicted: Virginica"),
            legend_point(CORRECT_LEGEND_SIZE, CORRECT_OPACITY, "Correct prediction"),
            legend_point(INCORRECT_LEGEND_SIZE, 1.0, "Wrong prediction"),
        ]

        plt.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left")
        plt.xlabel(f"{part} Length (cm)")
        plt.ylabel(f"{part} Width (cm)")
        plt.title(f"Species Predicted by GP, Plotted by {part} Dimensions")
        plt.show()

    def plot_confusion_matrix(self):
        cm = confusion_matrix(
            self.test_results["Species"], self.test_results["PredictedSpecies"]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        )
        disp.plot(cmap=plt.cm.RdPu)
        plt.xlabel("Predicted Species")
        plt.ylabel("Actual Species")
        plt.title("GP Predicted Species Confusion Matrix")
        plt.show()

    @staticmethod
    def plot_paramter_sweep_results(param_grid: dict, df: pd.DataFrame):
        """
        Plot the results of the parameter sweep.
        Args:
            param_grid: Dictionary of parameters to sweep keys include population_size, generations,
            crossover_rate, mutation_rate, num_champions_to_survive and the values are the
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
            plt.show()
