from parse_tree import *
from genetic_operators import GeneticOperators
import pandas as pd
import random
from copy import deepcopy


class IrisGP:
    """
    Classifier for the Iris dataset using Genetic Programming.

    Attributes:
        function_set (list[str]): The set of functions for parse trees to use.
        terminal_rules (TerminalGenerationRules): The rules for generating
            terminals, a set of literals, and a range for generating random
            constants.
        depth (int): The maximum depth of the tree. Expected to be 1 or greater
        terminal_prob (float): The probability of a node being a terminal node.
            Expected to be between 0 and 1.
    """

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
    def evaluate_fitness(self, individual: ParseTree, train_df: pd.DataFrame) -> int:
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
            fitness += self.evaluate_row(individual, row)
        return fitness

    def evaluate_row(self, individual: ParseTree, row: pd.Series, threshold=0.5) -> int:
        """
        Checks if the individual (parse tree) classifies a row correctly.

        The classification is based on comparing the output of the parse tree
        to a threshold value:
            output < threshold --> Iris setosa
            output >= threshold --> Iris virginica

        This function assumes the variables in the parse tree have the same
        names as the columns in the row.

        Args:
            individual (ParseTree): The parse tree to evaluate.
            row (pd.Series): A row of the training dataset, representing
                various measurements of an iris flower.
            threshold (float): The threshold value for classification. Defaults to 0.5.

        Returns (int): 1 if the classification is correct, 0 otherwise.
        """
        res = individual.evaluate(dict(row))
        if (res < threshold) == (row["Species"] == "Iris-setosa"):
            return 1
        else:
            return 0

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
        num_parents_to_survive: int,  # number of parents that move on to the generation
        train_df: pd.DataFrame,
    ) -> tuple[ParseTree, float, list[float]]:
        """
        Perform genetic programming, evolving a parse tree to classify the Iris
        dataset.

        Args:
            population_size (int): The number of individuals in the population.
            generations (int): The max number of generations to evolve for.
            crossover_rate (float): The probability of crossover between parents.
            mutation_rate (float): The probability of mutation in the offspring.
            num_parents_to_survive (int): The number of parents that move on to
                the next generation.
            train_df (pd.DataFrame): The portion of the dataset for training.

        Returns (tuple):
            - best_individual (ParseTree): The best parse tree found.
            - best_fitness (float): The fitness score of the best individual.
            - fitness_history (list[float]): A history of fitness scores over generations.
        """

        class FitnessCache:
            """
            Stores fitness scores of individuals to avoid recalculating.
            """

            def __init__(self):
                # Initialize None to -inf so a best individual of None is always overridden
                self.map = {None: float("-inf")}

            def __getitem__(self, key: ParseTree) -> float:
                """
                Returns the fitness score for a given individual.
                Evaluates the fitness if not already calculated.

                Args:
                    key (list): The schedule to evaluate

                Returns (float): The fitness score of the individual
                """
                if key not in self.map:
                    self.map[key] = IrisGP.evaluate_fitness(key, train_df)
                return self.map[key]

        population = self.generate_population(population_size)
        best_individual = None
        fitness_history = []
        fitness_cache = FitnessCache()

        for _ in range(generations):
            parents: list[ParseTree]  # select `num_parents_to_survive` best parents
            offspring: list[ParseTree] = []

            for _ in range(population_size - num_parents_to_survive):
                # Spawn children
                parent1, parent2 = self.select_parents(population)
                child = GeneticOperators.crossover(
                    parent1, parent2
                )  # TODO: factor in crossover rate
                offspring.append(child)

            population = parents + offspring

            # Mutate the population
            for idx, individual in enumerate(population):
                population[idx] = GeneticOperators.subtree_mutation(
                    individual,
                    self.function_set,
                    self.terminal_rules,
                    self.max_depth,
                )  # TODO: factor in mutation rate

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
