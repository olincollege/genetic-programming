from parse_tree import *
from genetic_operators import GeneticOperators
import pandas as pd
import random
from copy import deepcopy


class GeneticProgramming:
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
        fitness = 0
        for _, row in train_df.iterrows():
            fitness += self.evaluate_row(individual, row)
        return fitness

    def evaluate_row(self, individual: ParseTree, row: pd.Series) -> int:
        # < THRESHOLD --> setosa
        # >= THRESHOLD --> viriginica
        THRESHOLD = 0.5

        res = individual.evaluate(dict(row))
        # print(res, row["Species"], end=" ")
        if (res < THRESHOLD) == (row["Species"] == "Iris-setosa"):
            # print("1")
            return 1
        else:
            # print("0")
            return 0

    def generate_population(self, population_size: int) -> list[ParseTree]:
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
        self, population: list[ParseTree]
    ) -> tuple[ParseTree, ParseTree]:
        """
        Roulette wheel parent selection
        """
        pass

    def solve(
        self,
        population_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        num_parents_to_survive: int,  # number of parents that move on to the generation
        train_df: pd.DataFrame,
    ) -> tuple[ParseTree, float, list[float]]:
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
                    self.map[key] = GeneticProgramming.evaluate_fitness(key, train_df)
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
