from sklearn.metrics import accuracy_score
import pandas as pd
from genetic_programming import IrisGP

DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 20
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CHAMPION_SURVIVAL_PERCENTAGE = 0.1


def sweep_all_parameters(
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
        crossover_rate, mutation_rate, champion_survival_percentage and the values are the
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
    out = pd.DataFrame()
    for param_name, values in param_grid.items():
        results = sweep_single_parameter(
            param_name,
            values,
            train_df,
            test_df,
            FUNCTION_SET,
            TERMINAL_RULES,
            MAX_DEPTH,
            TERMINAL_PROBABILITY,
        )
        out = pd.concat([out, results])
    return out


def sweep_single_parameter(
    param_name,
    param_values,
    train_df,
    test_df,
    FUNCTION_SET,
    TERMINAL_RULES,
    MAX_DEPTH,
    TERMINAL_PROBABILITY,
):
    """
    Sweep through a single parameter and evaluate the GP model.
    Args:
        param_name: Name of the parameter to sweep.
        param_values: List of values for the parameter.
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
        "population_size": DEFAULT_POPULATION_SIZE,
        "generations": DEFAULT_GENERATIONS,
        "crossover_rate": DEFAULT_CROSSOVER_RATE,
        "mutation_rate": DEFAULT_MUTATION_RATE,
        "champion_survival_percentage": DEFAULT_CHAMPION_SURVIVAL_PERCENTAGE,
    }

    print(f"Sweeping {param_name}...")
    for val in param_values:
        print(f"{param_name} = {val}")
        params = base_params.copy()
        params[param_name] = val

        accuracies = []
        for _ in range(3):  # multiple trials for averaging
            gp = IrisGP(FUNCTION_SET, TERMINAL_RULES, MAX_DEPTH, TERMINAL_PROBABILITY)
            best_tree, _, _ = gp.solve(
                population_size=params["population_size"],
                generations=params["generations"],
                crossover_rate=params["crossover_rate"],
                mutation_rate=params["mutation_rate"],
                champion_survival_percentage=params["champion_survival_percentage"],
                train_df=train_df,
            )
            preds = [
                IrisGP.tree_to_class(best_tree, row) for _, row in test_df.iterrows()
            ]
            acc = accuracy_score(test_df["Species"], preds)
            accuracies.append(acc)
            print(f"Accuracy: {acc}")

        avg_acc = sum(accuracies) / len(accuracies)
        results.append({"param": param_name, "value": val, "accuracy": avg_acc})

    # Convert to DataFrame for plotting
    df = pd.DataFrame(results)
    return df
