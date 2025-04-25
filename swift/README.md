# Genetic Programming in Swift

A Swift implementation of genetic programming for evolving solutions to classification problems, using the Iris dataset.

## Code Organization

- **Data Structures**
  - `IrisFeatures`, `IrisSample`: Represent the Iris dataset
  - `IrisDataset`: Manages data loading, normalization, and splitting

- **Genetic Programming Core**
  - `ParseNode` (protocol): Base for all tree nodes
  - `TerminalNode`: Leaf nodes (variables/constants)
  - `FunctionNode`: Internal nodes (operations)
  - `ParseTree`: Complete expression trees
  - `TerminalGenerationRules`: Controls terminal node creation

- **Genetic Algorithm**
  - `GeneticProgramming`: Manages the evolutionary process
  - Fitness evaluation, selection, crossover, and mutation

## Key Features

- Tree-based representation with mathematical operators (+, -, *, /)
- "One-vs-all" approach for multi-class classification
- Tournament selection for parent choice
- Elitism to preserve best solutions
- Configurable mutation and crossover rates

## Implementation Highlights

- Protected division to prevent runtime errors (returns 1.0 when dividing by zero)
- Both "Full" and "Grow" methods for initial tree generation
- Subtree mutation replaces random nodes with new subtrees
- Deep copy implementation for tree manipulation without side effects
- Data normalization to improve learning

## Known Issues

- Fitness scores remain unchanged across generations, indicating optimization issues
- Potential inefficiencies in mutation or crossover operations

## Running the Code

```swift
swift main.swift
```

The program automatically loads the Iris dataset, trains classifiers for each species, and reports test accuracy.
