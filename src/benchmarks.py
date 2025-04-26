"""
Benchmarking module for genetic programming performance comparison.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Tuple, Optional, Any
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing modules with the appropriate paths
from parse_tree import ParseTree, TerminalGenerationRules, FunctionNode, TerminalNode
from genetic_operators import GeneticOperators

# For demonstration, let's also create a simpler data loader
# If you already have this in another module, you can import it instead
def load_iris_data():
    """Load and preprocess the Iris dataset"""
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("Required libraries not found. Install using:")
        print("pip install pandas scikit-learn matplotlib")
        return None, None, None, None
    
    # Load iris dataset
    iris_data = pd.read_csv("../data/Iris.csv")
    
    # Extract features and convert species to numeric
    X = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
    
    # Map species to numeric values
    species_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    y = iris_data['Species'].map(species_map).values
    
    # Normalize features
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


class IrisGP:
    """Genetic Programming for Iris classification"""
    
    def __init__(self, 
                 function_set=None, 
                 terminal_rules=None,
                 population_size=100,
                 generations=20,
                 mutation_rate=0.1,
                 crossover_rate=0.9,
                 max_depth=4,
                 terminal_prob=0.3,
                 num_parents_to_survive=10):
        
        self.function_set = function_set or ["+", "-", "*", "/"]
        self.terminal_rules = terminal_rules or TerminalGenerationRules(
            ["X1", "X2", "X3", "X4"], (-10, 10), ints_only=False
        )
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.terminal_prob = terminal_prob
        self.num_parents_to_survive = num_parents_to_survive
        
        self.population = []
        self.fitness_history = []
        self.best_tree = None
        self.best_fitness = 0
        
    def initialize_population(self):
        """Initialize the population with random trees"""
        self.population = []
        
        # Half full, half grow method
        for i in range(self.population_size // 2):
            tree = ParseTree.generate_full(
                self.function_set, self.terminal_rules, self.max_depth
            )
            self.population.append(tree)
            
        for i in range(self.population_size - self.population_size // 2):
            tree = ParseTree.generate_grow(
                self.function_set, self.terminal_rules, self.max_depth, self.terminal_prob
            )
            self.population.append(tree)
    
    def fitness_function(self, tree, X, y, target_class=-1):
        """Fitness function for Iris classification"""
        correct_count = 0
        
        for i in range(len(X)):
            features = X[i]
            actual_label = y[i]
            
            # Create variable values mapping
            variable_values = {
                "X1": features[0],
                "X2": features[1],
                "X3": features[2],
                "X4": features[3]
            }
            
            # Evaluate the tree to get predicted value
            predicted_value = tree.evaluate(variable_values)
            
            # Binary classification for one-vs-all
            if target_class >= 0:
                predicted_class = target_class if predicted_value > -0.5 else \
                  (1 if target_class == 0 else 0)
                if predicted_class == actual_label:
                    correct_count += 1
            else:
                # Simple classification based on value range
                if predicted_value < 0.33:
                    predicted_class = 0  # Setosa
                elif predicted_value < 0.66:
                    predicted_class = 1  # Versicolor
                else:
                    predicted_class = 2  # Virginica
                    
                if predicted_class == actual_label:
                    correct_count += 1
        
        return correct_count / len(X)
    
    def tournament_selection(self, fitnesses, tournament_size=7):
        """Tournament selection for parent choice"""
        best_index = np.random.randint(0, self.population_size)
        best_fitness = fitnesses[best_index]
        
        for _ in range(1, tournament_size):
            index = np.random.randint(0, self.population_size)
            fitness = fitnesses[index]
            
            if fitness > best_fitness:
                best_index = index
                best_fitness = fitness
        
        return self.population[best_index]
    
    def evolve(self, X, y, target_class=-1):
        """Evolve the population for a number of generations"""
        start_time = time.time()
        
        self.initialize_population()
        
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitnesses = [self.fitness_function(tree, X, y, target_class) for tree in self.population]
            
            # Find the best individual
            max_index = np.argmax(fitnesses)
            best = fitnesses[max_index]
            
            # Update best individual if improved
            if best > self.best_fitness:
                self.best_fitness = best
                self.best_tree = self.population[max_index]
            
            # Track average fitness
            avg_fitness = np.mean(fitnesses)
            self.fitness_history.append(avg_fitness)
            
            # Print progress every 5 generations
            if generation % 5 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}, Avg fitness = {avg_fitness:.4f}")
            
            # Create a new population
            new_population = []
            
            # Elitism - keep the best individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.num_parents_to_survive):
                new_population.append(self.population[sorted_indices[i]])
            
            # Fill the rest through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(fitnesses)
                
                if np.random.random() < self.crossover_rate:
                    # Crossover
                    parent2 = self.tournament_selection(fitnesses)
                    child1, _ = GeneticOperators.crossover(parent1, parent2)
                    
                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        child1 = GeneticOperators.subtree_mutation(
                            child1, self.function_set, self.terminal_rules, self.max_depth
                        )
                    
                    new_population.append(child1)
                else:
                    # No crossover, just mutation
                    if np.random.random() < self.mutation_rate:
                        new_population.append(GeneticOperators.subtree_mutation(
                            parent1, self.function_set, self.terminal_rules, self.max_depth
                        ))
                    else:
                        # Direct copy
                        new_population.append(parent1)
            
            # Ensure we don't exceed population size
            self.population = new_population[:self.population_size]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nEvolution completed in {execution_time:.2f} seconds")
        print(f"Best fitness: {self.best_fitness:.4f}")
        
        return self.best_tree, execution_time
    
    def solve_multi_class(self, X, y, num_classes=3):
        """Multi-class classification using one-vs-all approach"""
        classifiers = []
        execution_times = []
        
        # Train one classifier per class
        for class_index in range(num_classes):
            print(f"\nTraining classifier for class {class_index}")
            classifier, exec_time = self.evolve(X, y, target_class=class_index)
            classifiers.append(classifier)
            execution_times.append(exec_time)
            print(f"Best fitness for class {class_index}: {self.best_fitness:.4f}")
            
            # Reset for next class
            self.best_fitness = 0
            self.best_tree = None
        
        total_time = sum(execution_times)
        print(f"\nTotal training time: {total_time:.2f} seconds")
        
        return classifiers, total_time
    
    def test_multi_class(self, classifiers, X_test, y_test):
        """Test multi-class classification accuracy"""
        predictions = []
        
        for i in range(len(X_test)):
            features = X_test[i]
            
            # Create variable values mapping
            variable_values = {
                "X1": features[0],
                "X2": features[1],
                "X3": features[2],
                "X4": features[3]
            }
            
            # Get confidence scores from each classifier
            confidences = [
                classifier.evaluate(variable_values) for classifier in classifiers
            ]
            
            # Predict class with highest confidence
            predicted_class = np.argmax(confidences)
            predictions.append(predicted_class)
        
        # Calculate accuracy
        accuracy = np.mean(np.array(predictions) == np.array(y_test))
        print(f"\nTest accuracy: {accuracy * 100:.2f}%")
        
        # Print classification report
        try:
            from sklearn.metrics import classification_report
            print("\nClassification Report:")
            print(classification_report(
                          y_test, predictions, 
                          target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
                          zero_division=0))
        except ImportError:
            print("sklearn not available for detailed report")
        
        return accuracy


def run_benchmarks():
    """Run performance benchmarks for Python GP implementation"""
    print("Running Python GP benchmarks...\n")
    
    # Load iris dataset
    X_train, X_test, y_train, y_test = load_iris_data()
    if X_train is None:
        print("Failed to load dataset. Exiting.")
        return None
    
    # Configuration for benchmarks
    population_sizes = [50, 100, 200]
    generation_counts = [10, 20, 50]
    
    results = []
    
    # Fixed parameters
    fixed_params = {
        'mutation_rate': 0.1,
        'crossover_rate': 0.9,
        'max_depth': 4,
        'terminal_prob': 0.3
    }
    
    # Run benchmarks with varying population size (smaller for faster testing)
    print("\nBenchmarking with varying population size...")
    for pop_size in population_sizes:
        gp = IrisGP(population_size=pop_size, generations=10, **fixed_params)
        start_time = time.time()
        classifiers, _ = gp.solve_multi_class(X_train, y_train)
        end_time = time.time()
        total_time = end_time - start_time
        
        accuracy = gp.test_multi_class(classifiers, X_test, y_test)
        
        results.append({
            'parameter': 'Population Size',
            'value': pop_size,
            'time': total_time,
            'accuracy': accuracy
        })
        
        print(f"Population size {pop_size}: {total_time:.2f} seconds, Accuracy: {accuracy:.4f}")
    
    # Run benchmarks with varying generations (smaller for faster testing)
    print("\nBenchmarking with varying generation count...")
    for gen_count in generation_counts:
        gp = IrisGP(population_size=50, generations=gen_count, **fixed_params)
        start_time = time.time()
        classifiers, _ = gp.solve_multi_class(X_train, y_train)
        end_time = time.time()
        total_time = end_time - start_time
        
        accuracy = gp.test_multi_class(classifiers, X_test, y_test)
        
        results.append({
            'parameter': 'Generations',
            'value': gen_count,
            'time': total_time,
            'accuracy': accuracy
        })
        
        print(f"Generation count {gen_count}: {total_time:.2f} seconds, Accuracy: {accuracy:.4f}")
    
    # Save results to CSV
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv('../data/python_gp_benchmarks.csv', index=False)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot population size benchmarks
        pop_results = results_df[results_df['parameter'] == 'Population Size']
        plt.subplot(1, 2, 1)
        plt.plot(pop_results['value'], pop_results['time'], 'o-', label='Execution Time')
        plt.xlabel('Population Size')
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time vs Population Size')
        plt.grid(True)
        
        # Plot generation benchmarks
        gen_results = results_df[results_df['parameter'] == 'Generations']
        plt.subplot(1, 2, 2)
        plt.plot(gen_results['value'], gen_results['time'], 'o-', label='Execution Time')
        plt.xlabel('Generations')
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time vs Generations')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../docs/img/python_benchmarks.png')
        plt.close()
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
    
    return results


if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    if benchmark_results is not None:
        print("\nBenchmark results saved to '../data/python_gp_benchmarks.csv'")
        print("Benchmark plots saved to '../data/python_benchmarks.png'")
