"""
Genetic Programming for the Iris Dataset.
"""

import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from genetic_programming import IrisGP
from parse_tree import TerminalGenerationRules


def main():
    """
    Main function to run the Genetic Programming for Iris dataset classification.
    """
    # Set random seed
    random.seed(2)

    # Load the full Iris dataset
    iris = pd.read_csv("data/Iris.csv")

    # Split into train and test
    train_df, test_df = train_test_split(iris, test_size=0.2, random_state=3)

    # Define GP components
    function_set = ["+", "-", "*", "/"]
    terminal_rules = TerminalGenerationRules(
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        (-10, 10),
        ints_only=False,
        no_random_constants=False,
    )

    # Instantiate the GP system
    gp = IrisGP(
        function_set=function_set,
        terminal_rules=terminal_rules,
        max_depth=3,
        terminal_prob=0.3,
    )

    # Train the model
    best_tree, _, _ = gp.solve(
        population_size=50,
        generations=20,
        crossover_rate=0.9,
        mutation_rate=0.1,
        champion_survival_percentage=0.1,
        train_df=train_df,
    )

    # Use the trained tree to classify test instances
    def predict_species(parse_tree, row):
        """
        Predict the species of an Iris flower using the parse tree.

        Args:
            parse_tree: The parse tree representing the model.
            row: A row from the test DataFrame.
        Returns:
            str: The predicted species.
        """
        prediction = parse_tree.evaluate(row)
        # Map prediction to the closest class
        if prediction < IrisGP.THRESHOLD_LOW:
            return "Iris-setosa"
        elif prediction < IrisGP.THRESHOLD_HIGH:
            return "Iris-versicolor"
        else:
            return "Iris-virginica"

    # Evaluate
    y_true = test_df["Species"].tolist()
    y_pred = [predict_species(best_tree, row) for _, row in test_df.iterrows()]
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
