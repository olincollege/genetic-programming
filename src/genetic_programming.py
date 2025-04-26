"""
Genetic Programming for the Iris Dataset.
"""

import random
from copy import deepcopy
import pandas as pd
from parse_tree import ParseTree, FunctionNode, TerminalNode, TerminalGenerationRules


class IrisGP:
    """
    Classifier for the Iris dataset using Genetic Programming.

    Constants:
        THRESHOLD_LOW (float): The low threshold for classification.
        THRESHOLD_HIGH (float): The high threshold for classification.

        Classification is based on comparing the output of the parse tree
        to the thresholds:
            res < THRESHOLD_LOW                     --> Iris-setosa
            THRESHOLD_LOW <= res < THRESHOLD_HIGH   --> Iris-versicolor
            res >= THRESHOLD_HIGH                   --> Iris-virginica

    Attributes:
        function_set (list[str]): The set of functions for parse trees to use.
        terminal_rules (TerminalGenerationRules): The rules for generating
            terminals, a set of literals, and a range for generating random
            constants.
        depth (int): The maximum depth of the tree. Expected to be 1 or greater
        terminal_prob (float): The probability of a node being a terminal node.
            Expected to be between 0 and 1.
    """

    THRESHOLD_LOW = 0.33
    THRESHOLD_HIGH = 0.66

    def __init__(
        self,
        function_set: list,
        terminal_rules: TerminalGenerationRules,
        max_depth: int,
        terminal_prob: float,
    ):
        self.function_set = function_set
        self.terminal_rules = terminal_rules
        self.max_depth = max_depth
        self.terminal_prob = terminal_prob

    @staticmethod
    def evaluate_fitness(individual: ParseTree, train_df: pd.DataFrame) -> int:
        """
        Evaluates the fitness of an individual (parse tree) by the number of
        correctly classified rows in the training dataset.

        Args:
            individual (ParseTree): The parse tree to evaluate.
            train_df (pd.DataFrame): The training Iris dataset.

        Returns (int): The fitness score of the individual.
        """
        fitness = 0
        for _, row in train_df.iterrows():
            correctness, _ = IrisGP.evaluate_row(individual, row)
            fitness += correctness
        return fitness

    @staticmethod
    def evaluate_row(individual: ParseTree, row: pd.Series) -> tuple[bool, float]:
        """
        Checks if the individual (parse tree) classifies a row correctly.
        Classification is based comparing the output of the parse tree
        to the thresholds (see class docstring above).

        This function assumes the variables in the parse tree have the same
        names as the columns in the row.

        Args:
            individual (ParseTree): The parse tree to evaluate.
            row (pd.Series): A row of the training dataset, representing
                various measurements of an iris flower.
            t1 (float): The threshold value for classification. Defaults to 0.33.
            t2 (float): The threshold value for classification. Defaults to 0.66.

        Returns (tuple):
            bool: True if the classification is correct, False otherwise.
            float: The output of the parse tree for the given row.
        """
        res = individual.evaluate(dict(row))
        if res < IrisGP.THRESHOLD_LOW:
            predicted = "Iris-setosa"
        elif res < IrisGP.THRESHOLD_HIGH:
            predicted = "Iris-versicolor"
        else:
            predicted = "Iris-virginica"

        correct = predicted == row["Species"]
        return correct, res

    def generate_population(self, population_size: int) -> list[ParseTree]:
        """
        Generates an initial population of parse trees.

        Args:
            population_size (int): The size of the population to generate.

        Returns (list[ParseTree]): A list of randomly generated parse trees.
        """
        # TODO: make optional grow vs full
        return [
            ParseTree.generate_grow(
                self.function_set,
                self.terminal_rules,
                self.max_depth,
                self.terminal_prob,
            )
            for _ in range(population_size)
        ]

    def select_parents(
        self, population: list[ParseTree], fitness_cache: dict[ParseTree, float]
    ) -> tuple[ParseTree, ParseTree]:
        """
        Select two parents via roulette wheel parent selection. Parents are
        randomly selected with probability proportional to their fitness.

        Args:
            population (list): The current population of individuals.
            fitness_cache (FitnessCache): A cache of fitness scores for the individuals.

        Returns (tuple): A tuple of two selected parents.
        """
        fitnesses = [fitness_cache[i] for i in population]
        return random.choices(population, weights=fitnesses, k=2)

    def solve(
        self,
        population_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        num_champions_to_survive: int,
        train_df: pd.DataFrame,
    ) -> tuple[ParseTree, float, list[float]]:
        """
        Perform genetic programming, evolving a parse tree to classify the Iris
        dataset.

        Args:
            population_size (int): The number of individuals in the population.
            generations (int): The max number of generations to evolve for.
            crossover_rate (float): The probability of crossover between parents
                from zero to one.
            mutation_rate (float): The probability of mutation in the offspring
                from zero to one.
            num_champions_to_survive (int): The number of best performing
                individuals that move onto the next generation.
            train_df (pd.DataFrame): The portion of the dataset for training.

        Returns (tuple):
            - best_individual (ParseTree): The best parse tree found.
            - best_fitness (float): The fitness score of the best individual.
            - fitness_history (list[float]): A history of fitness scores over generations.
        """

        population = self.generate_population(population_size)
        best_individual = None
        fitness_history = []
        fitness_cache = FitnessCache(train_df)

        for _ in range(generations):
            # Let n best individuals survive to the next generation as parents
            champions: list[ParseTree] = sorted(
                population, key=lambda x: fitness_cache[x], reverse=True
            )[:num_champions_to_survive]
            offspring: list[ParseTree] = []

            for _ in range(population_size - num_champions_to_survive):
                # Spawn children
                parent1, parent2 = self.select_parents(population, fitness_cache)
                if random.random() < crossover_rate:
                    # `crossover_rate` is the probability of crossover
                    child, _ = GeneticOperators.crossover(parent1, parent2)
                    offspring.append(deepcopy(child))
                else:
                    # If no crossover, just clone one of the parents
                    offspring.append(deepcopy(parent1))

            population = champions + offspring

            # Mutate the population
            for idx, individual in enumerate(population):
                # `mutation_rate` chance to mutate each individual
                if random.random() > mutation_rate:
                    continue
                population[idx] = GeneticOperators.subtree_mutation(
                    individual,
                    self.function_set,
                    self.terminal_rules,
                    self.max_depth,
                    self.terminal_prob,
                )

            # Update History
            fitness_history.append(
                sum([fitness_cache[i] for i in population]) / len(population)
            )

            # Check for best individual
            generation_best = max(population, key=lambda x: fitness_cache[x])
            if fitness_cache[generation_best] > fitness_cache[best_individual]:
                best_individual = deepcopy(generation_best)
                # A fitness of len(train_df) means all rows were classified correctly
                if fitness_cache[best_individual] == len(train_df):
                    break

        return best_individual, fitness_cache[best_individual], fitness_history

    @staticmethod
    def tree_to_class(tree, row):
        """
        Convert the parse tree to a class label based on the output value.
        Args:
            tree: The parse tree representing the model.
            row: A row from the test DataFrame.
        Returns:
            The predicted class label. Should be one of "Iris-setosa", "Iris-versicolor",
            or "Iris-virginica".
        """
        value = tree.evaluate(row)
        class_index = round(value)
        class_index = max(0, min(2, class_index))  # clamp to 0–2
        return ["Iris-setosa", "Iris-versicolor", "Iris-virginica"][class_index]


class FitnessCache:
    """
    Stores fitness scores of individuals to avoid recalculating.

    Attributes:
        map (dict): A dictionary mapping individuals to their fitness scores.
        train_df (pd.DataFrame): The training dataset used for fitness evaluation.
    """

    def __init__(self, train_df: pd.DataFrame):

        self.map = {}
        self.train_df = train_df

    def __getitem__(self, tree: ParseTree) -> float:
        """
        Returns the fitness score for a given individual.
        Evaluates the fitness if not already calculated.

        Args:
            tree (ParseTree): The tree to get the fitness of.

        Returns (float): The fitness score of the individual
        """
        # Map None to -inf so a best individual of None is always overridden
        if tree is None:
            return float("-inf")

        # Could change this to `key = repr(tree)` for more correct hashing,
        # however it slows down the runtime by 3x, so this works good enough.
        key = tree
        if key not in self.map:
            self.map[key] = IrisGP.evaluate_fitness(tree, self.train_df)
        return self.map[key]


class GeneticOperators:
    """
    A class containing various genetic operators for tree-based genetic programming.
    """

    @staticmethod
    def subtree_mutation(
        tree: "ParseTree",
        function_set: list[str],
        terminal_rules: "TerminalGenerationRules",
        max_depth: int,
        terminal_prob: float,
    ) -> "ParseTree":
        """
        Replace a random subtree in the tree with a new subtree generated from the function set.

        Args:
            tree (ParseTree): The parse tree to mutate.
            function_set (list): The set of functions to use for mutation.
            terminal_rules (TerminalGenerationRules): The rules for generating terminal nodes.
            max_depth (int): The maximum depth of the new subtree.
            terminal_prob (float): The probability of a node being a terminal
                node when generating a new subtree.

        Returns:
            ParseTree: The mutated parse tree.
        """
        mutated_tree = deepcopy(tree)
        node_to_replace, parent = mutated_tree.get_random_node()
        new_subtree = FunctionNode.from_function_set(
            function_set, terminal_rules, max_depth, terminal_prob=0.1
        )

        if parent is None:
            mutated_tree.root = new_subtree
        else:
            for i, child in enumerate(parent.children):
                if child is node_to_replace:
                    parent.children[i] = new_subtree
                    break

        return mutated_tree

    @staticmethod
    def leaf_replacement(
        tree: "ParseTree", terminal_rules: "TerminalGenerationRules"
    ) -> "ParseTree":
        """
        Replace a random leaf node in the tree with a new terminal node.
        Args:
            tree (ParseTree): The parse tree to mutate.
            terminal_rules (TerminalGenerationRules): The rules for generating terminal nodes.

        Returns:
            ParseTree: The mutated parse tree.
        """
        mutated_tree = deepcopy(tree)
        leaf_node, parent = mutated_tree.get_random_node("leaf")

        new_leaf = TerminalNode.from_terminal_set(terminal_rules)

        if parent is None:
            mutated_tree.root = new_leaf
        else:
            # Find and replace the leaf node with the new terminal node
            for i, child in enumerate(parent.children):
                if child is leaf_node:
                    parent.children[i] = new_leaf
                    break

        return mutated_tree

    @staticmethod
    def crossover(
        tree1: "ParseTree", tree2: "ParseTree"
    ) -> tuple["ParseTree", "ParseTree"]:
        """
        Perform crossover between two parse trees.
        Selects a random node from each tree and swaps their subtrees.

        Args:
            tree1 (ParseTree): The first parse tree.
            tree2 (ParseTree): The second parse tree.

        Returns:
            tuple: The modified parse trees after crossover.
        """
        # Select a random node from each tree
        node1, parent1 = tree1.get_random_node()
        node2, parent2 = tree2.get_random_node()
        # Ensure the nodes are not the same
        while node1 is node2:
            node2, parent2 = tree2.get_random_node()
        # Swap the subtrees
        if parent1 is None:
            tree1.root = deepcopy(node2)
        else:
            for i, child in enumerate(parent1.children):
                if child is node1:
                    parent1.children[i] = deepcopy(node2)
                    break
        if parent2 is None:
            tree2.root = deepcopy(node1)
        else:
            for i, child in enumerate(parent2.children):
                if child is node2:
                    parent2.children[i] = deepcopy(node1)
                    break
        return tree1, tree2
