from sklearn.metrics import accuracy_score
import pandas as pd
from genetic_programming import IrisGP


def sweep_parameters(
    param_grid,
    train_df,
    test_df,
    FUNCTION_SET,
    TERMINAL_RULES,
    MAX_DEPTH,
    TERMINAL_PROBABILITY,
):
    """
    Sweep through the parameter grid and evaluate the GP model.
    Args:
        param_grid: Dictionary of parameters to sweep keys include population_size, generations,
        crossover_rate, mutation_rate, num_champions_to_survive and the values are the
        lists of the parameters.
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        FUNCTION_SET: Function set for the GP.
        TERMINAL_RULES: Terminal generation rules for the GP.
        MAX_DEPTH: Maximum depth for the GP trees.
        TERMINAL_PROBABILITY: Probability of generating a terminal node.
    Returns:
        DataFrame with results.
    """
    # Initialize results list
    results = []

    # Base/default values
    base_params = {
        "population_size": 100,
        "generations": 20,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "num_champions_to_survive": 10,
    }

    for param_name, values in param_grid.items():
        for val in values:
            params = base_params.copy()
            params[param_name] = val

            accuracies = []
            for _ in range(3):  # multiple trials for averaging
                gp = IrisGP(
                    FUNCTION_SET, TERMINAL_RULES, MAX_DEPTH, TERMINAL_PROBABILITY
                )
                best_tree, _, _ = gp.solve(
                    population_size=params["population_size"],
                    generations=params["generations"],
                    crossover_rate=params["crossover_rate"],
                    mutation_rate=params["mutation_rate"],
                    num_champions_to_survive=params["num_champions_to_survive"],
                    train_df=train_df,
                )
                preds = [
                    IrisGP.tree_to_class(best_tree, row)
                    for _, row in test_df.iterrows()
                ]
                acc = accuracy_score(test_df["Species"], preds)
                accuracies.append(acc)

            avg_acc = sum(accuracies) / len(accuracies)
            results.append({"param": param_name, "value": val, "accuracy": avg_acc})

    # Convert to DataFrame for plotting
    df = pd.DataFrame(results)
    return df
