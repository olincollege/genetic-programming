import Foundation

// MARK: - Iris Data Structures

/// A structure representing the four measurements of an iris flower.
struct IrisFeatures {
    /// Length of the sepal in centimeters
    var sepalLength: Double

    /// Width of the sepal in centimeters
    var sepalWidth: Double

    /// Length of the petal in centimeters
    var petalLength: Double

    /// Width of the petal in centimeters
    var petalWidth: Double
}

/// A complete iris sample with both features and classification information.
struct IrisSample {
    /// The four measurements of the iris flower
    let features: IrisFeatures
    
    /// The species name (e.g., "Iris-setosa", "Iris-versicolor", "Iris-virginica")
    let species: String
    
    /// Numeric index of the species (0 = setosa, 1 = versicolor, 2 = virginica)
    let speciesIndex: Int
}

// MARK: - Iris Dataset Manager

/**
 * Manages loading, preprocessing, and partitioning of the Iris dataset.
 *
 * This class handles:
 * - Loading data from CSV file
 * - Data normalization
 * - Train/test splitting
 * - Conversion to formats needed for genetic programming
 */
class IrisDataset {
    /// Array of all iris samples loaded from the dataset
    private(set) var samples: [IrisSample] = []
    
    /// Array of normalized iris samples (features scaled to [0,1] range)
    private(set) var normalizedSamples: [IrisSample] = []

    /// Dictionary mapping species names to numeric indices
    private let speciesMap = [
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    ]
    
    /// Minimum values for each feature (used in normalization)
    private var minValues = IrisFeatures(sepalLength: Double.infinity, sepalWidth: Double.infinity, 
                                         petalLength: Double.infinity, petalWidth: Double.infinity)
    
    /// Maximum values for each feature (used in normalization)
    private var maxValues = IrisFeatures(sepalLength: -Double.infinity, sepalWidth: -Double.infinity, 
                                         petalLength: -Double.infinity, petalWidth: -Double.infinity)

    /**
     * Loads and parses the Iris dataset from a CSV file.
     
     - Parameters:
        - filePath: Path to the CSV file containing the iris dataset.
     
     - Returns: A collection of parsed Iris data entries.
     
     - Throws: An error if the file cannot be found, read, or if the CSV format is invalid.
     

     - Note: The CSV file is expected to contain the standard Iris dataset format with
             columns for sepal length, sepal width, petal length, petal width, and species.
     */
    func loadFromCSV(filePath: String) throws {
        let fileContents = try String(contentsOfFile: filePath, encoding: .utf8)
        let lines = fileContents.split(separator: "\n")
        
        // Skip header line
        for i in 1..<lines.count {
            let line = lines[i]
            let values = line.split(separator: ",")
            
            // Ensure the line has the expected number of fields
            guard values.count >= 6 else {
                print("Warning: Skipping malformed line: \(line)")
                continue
            }
            
            // Parse numeric values and species
            guard let sepalLength = Double(values[1]),
                  let sepalWidth = Double(values[2]),
                  let petalLength = Double(values[3]),
                  let petalWidth = Double(values[4]) else {
                print("Warning: Failed to parse numeric values in line: \(line)")
                continue
            }
            
            let species = String(values[5])
            guard let speciesIndex = speciesMap[species] else {
                print("Warning: Unknown species: \(species)")
                continue
            }
            
            // Create and add the sample
            let features = IrisFeatures(
                sepalLength: sepalLength,
                sepalWidth: sepalWidth,
                petalLength: petalLength,
                petalWidth: petalWidth
            )
            
            // Update min/max values for normalization
            updateMinMaxValues(features: features)
            
            // Create and add the sample
            let sample = IrisSample(
                features: features,
                species: species,
                speciesIndex: speciesIndex
            )
            
            samples.append(sample)
        }
        
        normalizeData()        
        print("Loaded \(samples.count) samples from Iris dataset")
    }
    
    /**
     * Updates the minimum and maximum feature values for normalization.
    
     - Parameters:
        - features: The iris features to update min/max values with
     */
    private func updateMinMaxValues(features: IrisFeatures) {
        // Update min values
        minValues.sepalLength = min(minValues.sepalLength, features.sepalLength)
        minValues.sepalWidth = min(minValues.sepalWidth, features.sepalWidth)
        minValues.petalLength = min(minValues.petalLength, features.petalLength)
        minValues.petalWidth = min(minValues.petalWidth, features.petalWidth)
        
        // Update max values
        maxValues.sepalLength = max(maxValues.sepalLength, features.sepalLength)
        maxValues.sepalWidth = max(maxValues.sepalWidth, features.sepalWidth)
        maxValues.petalLength = max(maxValues.petalLength, features.petalLength)
        maxValues.petalWidth = max(maxValues.petalWidth, features.petalWidth)
    }
    
    /**
     * Normalizes all iris samples to have features in the [0,1] range. 
       Uses the previously calculated min/max values to perform min-max scaling.
     
     - Returns: None
     */
    private func normalizeData() {
        normalizedSamples = samples.map { sample in
            let normalizedFeatures = IrisFeatures(
                sepalLength: normalize(sample.features.sepalLength, min: minValues.sepalLength, max: maxValues.sepalLength),
                sepalWidth: normalize(sample.features.sepalWidth, min: minValues.sepalWidth, max: maxValues.sepalWidth),
                petalLength: normalize(sample.features.petalLength, min: minValues.petalLength, max: maxValues.petalLength),
                petalWidth: normalize(sample.features.petalWidth, min: minValues.petalWidth, max: maxValues.petalWidth)
            )
            
            return IrisSample(
                features: normalizedFeatures,
                species: sample.species,
                speciesIndex: sample.speciesIndex
            )
        }
    }
    
    /**
     * Normalizes a single feature value to the [0,1] range.

     - Parameters:
        - value: The feature value to normalize
        - min: The minimum value of the feature
        - max: The maximum value of the feature
    
     - Returns: The normalized value in the range [0,1]
     */
    private func normalize(_ value: Double, min: Double, max: Double) -> Double {
        return min == max ? 0 : (value - min) / (max - min)
    }
    
    /**
     * Splits the normalized dataset into training and testing subsets.
     
     - Parameters:
        - trainRatio: Proportion of data to use for training (default is 0.7)

     - Returns: A tuple containing training and testing datasets
     */
    func splitData(trainRatio: Double = 0.7) -> (train: [IrisSample], test: [IrisSample]) {
        // Shuffle the data to ensure random distribution
        let shuffledData = normalizedSamples.shuffled()
        
        // Calculate split index
        let splitIndex = Int(Double(shuffledData.count) * trainRatio)
        
        // Split the data
        let trainData = Array(shuffledData[0..<splitIndex])
        let testData = Array(shuffledData[splitIndex..<shuffledData.count])
        
        return (train: trainData, test: testData)
    }
    
    /**
     * Converts the normalized dataset to feature arrays and label arrays.
    
     - Returns: A tuple containing two arrays:
        - features: 2D array of feature values
        - labels: 1D array of class labels (species indices)
     */
    func getDataArrays() -> (features: [[Double]], labels: [Int]) {
        let features = normalizedSamples.map { sample in
            return [
                sample.features.sepalLength,
                sample.features.sepalWidth,
                sample.features.petalLength,
                sample.features.petalWidth
            ]
        }
        
        let labels = normalizedSamples.map { $0.speciesIndex }
        
        return (features: features, labels: labels)
    }
    
    /**
     * Returns the feature names for the dataset.
    
     - Returns: An array of feature names
     */
    func getFeatureNames() -> [String] {
        return ["X1", "X2", "X3", "X4"] // Corresponding to the 4 Iris features
    }
}

// MARK: - Genetic Programming Structures

/**
 * Terminal generation rules for creating terminal nodes in the parse tree.
 *
 * This class defines the rules for generating terminal nodes, including literals,
 * constants range, decimal places, and whether to use random constants or not.
 */
class TerminalGenerationRules {
    let literals: [String]
    let constantsRange: (Double, Double)
    let decimalPlaces: Int
    let intsOnly: Bool
    let noRandomConstants: Bool
    
    /**
     * Initializes the terminal generation rules.
     
     - Parameters:
        - literals: Array of variable names (e.g., ["X1", "X2", "X3", "X4"]).
        - constantsRange: Range for generating random constants (min, max).
        - decimalPlaces: Number of decimal places for rounding constants.
        - intsOnly: If true, only integer constants will be generated.
        - noRandomConstants: If true, only literals will be used (no random constants).
     */
    init(literals: [String], 
         constantsRange: (Double, Double), 
         decimalPlaces: Int = 4, 
         intsOnly: Bool = false, 
         noRandomConstants: Bool = false) {
        self.literals = literals
        self.constantsRange = constantsRange
        self.decimalPlaces = decimalPlaces
        self.intsOnly = intsOnly
        self.noRandomConstants = noRandomConstants
    }
}

/**
 * Protocol defining the structure of a parse node in the parse tree.
 *
 * This protocol is implemented by both terminal and function nodes.
 */
protocol ParseNode: AnyObject, CustomStringConvertible {
    var value: String { get }
    func evaluate(variableValues: [String: Double]) -> Double
    func copy() -> ParseNode
}

/**
 * A terminal node in the parse tree, representing a variable or constant.
 */
class TerminalNode: ParseNode {
    let value: String
    
    /**
     * Initializes a terminal node with a given value.
     
     - Parameters:
        - value: The value of the terminal node (e.g., variable name or constant).
     */
    init(value: String) {
        self.value = value
    }
    
    /**
     * Generates a terminal node from a set of literals and constants.
     
     This method randomly selects a literal or generates a random constant
     based on the provided rules. It returns a new TerminalNode instance.
     
     - Parameters:
        - rules: The rules for generating terminal nodes.
     
     - Returns: A new TerminalNode instance.
     */
    static func fromTerminalSet(rules: TerminalGenerationRules) -> TerminalNode {
        let options = rules.noRandomConstants ? rules.literals.count - 1 : rules.literals.count
        let randomIndex = Int.random(in: 0...options)
        
        if randomIndex < rules.literals.count {
            // Select a literal
            return TerminalNode(value: rules.literals[randomIndex])
        } else {
            // Generate a random constant
            var constant = Double.random(in: rules.constantsRange.0...rules.constantsRange.1)
            if rules.intsOnly {
                constant = Double(Int(constant))
            } else {
                // Round to specified decimal places
                let multiplier = pow(10.0, Double(rules.decimalPlaces))
                constant = round(constant * multiplier) / multiplier
            }
            return TerminalNode(value: String(constant))
        }
    }
    
    /**
     * Evaluates the terminal node with given variable values.
     
     This method checks if the value is a variable or a constant and returns
     the corresponding value based on the provided variable values.
     
     - Parameters:
        - variableValues: A dictionary mapping variable names to their values.
     
     - Returns: The evaluated value of the terminal node.
     */
    func evaluate(variableValues: [String: Double]) -> Double {
        if let variableValue = variableValues[value] {
            return variableValue
        }
        
        if let numericValue = Double(value) {
            return numericValue
        }
        
        fatalError("Invalid terminal value: \(value)")
    }
    
    /**
     * Returns a string representation of the terminal node.
     
     This method provides a simple description of the terminal node, which is
     useful for debugging and logging purposes.
     
     - Returns: A string representation of the terminal node.
     */
    var description: String {
        return value
    }
    
    /**
     * Creates a copy of the terminal node.
     
     This method creates a new instance of the terminal node with the same value.
     
     - Returns: A new TerminalNode instance that is a copy of the original.
     */
    func copy() -> ParseNode {
        return TerminalNode(value: self.value)
    }
}

/**
 * A function node in the parse tree, representing a mathematical operation.
 */
class FunctionNode: ParseNode {
    let value: String
    let arity: Int
    var children: [ParseNode]
    
    /**
     * Initializes a function node with a given value and children.
     
     - Parameters:
        - value: The name of the function (e.g., "+", "-", "sin").
        - children: An array of child nodes (either function or terminal nodes).
     */
    init(value: String, children: [ParseNode]) {
        self.value = value
        self.arity = FunctionNode.arityMap(function: value)
        self.children = children
    }
    
    /**
     * Maps function names to their arity (number of arguments).
     
     This method returns the number of arguments required for a given function.
     
     - Parameters:
        - function: The name of the function.
     
     - Returns: The arity of the function.
     */
    static func arityMap(function: String) -> Int {
        switch function {
        case "+", "-", "*", "/":
            return 2
        case "sin", "cos", "exp", "ln":
            return 1
        default:
            fatalError("Unknown function: \(function)")
        }
    }
    
    /**
     * Generates a function node from a set of functions and terminal rules.
     
     This method creates a function node with a specified depth and probability
     of generating terminal nodes. It recursively generates child nodes based on
     the provided function set and terminal rules.
     
     - Parameters:
        - functionSet: Array of function names to be used in the parse tree.
        - terminalRules: The rules for generating terminal nodes.
        - depth: Maximum depth of the tree.
        - terminalProb: Probability of generating a terminal node.
     
     - Returns: A new FunctionNode instance representing the generated function node.
     */
    static func fromFunctionSet(
        functionSet: [String], 
        terminalRules: TerminalGenerationRules, 
        depth: Int, 
        terminalProb: Double
    ) -> FunctionNode {
        let randomFunctionIndex = Int.random(in: 0..<functionSet.count)
        let function = functionSet[randomFunctionIndex]
        let node = FunctionNode(value: function, children: [])
        
        var children: [ParseNode] = []
        for _ in 0..<node.arity {
            if depth <= 1 || Double.random(in: 0...1) < terminalProb {
                // Generate a terminal node
                children.append(TerminalNode.fromTerminalSet(rules: terminalRules))
            } else {
                // Generate a function node
                children.append(fromFunctionSet(
                    functionSet: functionSet,
                    terminalRules: terminalRules,
                    depth: depth - 1,
                    terminalProb: terminalProb
                ))
            }
        }
        
        node.children = children
        return node
    }
    
    /**
     * Evaluates the function node with given variable values.
     
     This method recursively evaluates the function node and its children
     based on the provided variable values. It returns the result of the
     evaluation.
     
     - Parameters:
        - variableValues: A dictionary mapping variable names to their values.
     
     - Returns: The result of evaluating the function node.
     */
    func evaluate(variableValues: [String: Double]) -> Double {
        let evaluatedChildren = children.map { $0.evaluate(variableValues: variableValues) }
        
        switch value {
        case "+":
            return evaluatedChildren[0] + evaluatedChildren[1]
        case "-":
            return evaluatedChildren[0] - evaluatedChildren[1]
        case "*":
            return evaluatedChildren[0] * evaluatedChildren[1]
        case "/":
            // Protected division
            if evaluatedChildren[1] == 0 {
                return 1.0
            }
            return evaluatedChildren[0] / evaluatedChildren[1]
        case "sin":
            return sin(evaluatedChildren[0])
        case "cos":
            return cos(evaluatedChildren[0])
        case "exp":
            return exp(evaluatedChildren[0])
        case "ln":
            if evaluatedChildren[0] <= 0 {
                return -1.0
            }
            return log(evaluatedChildren[0])
        default:
            fatalError("Unknown function: \(value)")
        }
    }
    
    /**
     * Returns a string representation of the function node.
     
     This method provides a simple description of the function node, which is
     useful for debugging and logging purposes.
     
     - Returns: A string representation of the function node.
     */
    var description: String {
        let args = children.map { String(describing: $0) }.joined(separator: " ")
        return "(\(value) \(args))"
    }
    
    /**
     * Creates a copy of the function node and its children.
     
     This method creates a deep copy of the function node, including all its
     children nodes. It is useful for creating new instances of the parse tree
     without modifying the original structure.
     
     - Returns: A new FunctionNode instance that is a copy of the original.
     */
    func copy() -> ParseNode {
        let copiedChildren = children.map { $0.copy() }
        return FunctionNode(value: self.value, children: copiedChildren)
    }
}

/**
 * A parse tree representing a mathematical expression.
 *
 * This class manages the structure of the parse tree, including the root node,
 * and provides methods for generating, evaluating, and mutating the tree.
 */
class ParseTree: CustomStringConvertible {
    var root: FunctionNode
    
    /**
     * Initializes a parse tree with a given root node.
     
     - Parameters:
        - root: The root node of the parse tree.
     */
    init(root: FunctionNode) {
        self.root = root
    }
    
    /**
     * Generates a parse tree using the full method.
     
     This method creates a parse tree with a specified maximum depth and
     no probability of generating terminal nodes.
     
     - Parameters:
        - functionSet: Array of function names to be used in the parse tree.
        - terminalRules: The rules for generating terminal nodes.
        - depth: Maximum depth of the tree.
     
     - Returns: A new ParseTree instance.
     */
    static func generateFull(
        functionSet: [String], 
        terminalRules: TerminalGenerationRules, 
        depth: Int
    ) -> ParseTree {
        let root = FunctionNode.fromFunctionSet(
            functionSet: functionSet,
            terminalRules: terminalRules,
            depth: depth,
            terminalProb: 0.0
        )
        return ParseTree(root: root)
    }
    
    /**
     * Generates a parse tree using the grow method.
     
     This method creates a parse tree with a specified maximum depth and
     a probability of generating terminal nodes.
     
     - Parameters:
        - functionSet: Array of function names to be used in the parse tree.
        - terminalRules: The rules for generating terminal nodes.
        - depth: Maximum depth of the tree.
        - terminalProb: Probability of generating a terminal node.
     
     - Returns: A new ParseTree instance.
     */
    static func generateGrow(
        functionSet: [String], 
        terminalRules: TerminalGenerationRules, 
        depth: Int, 
        terminalProb: Double
    ) -> ParseTree {
        let root = FunctionNode.fromFunctionSet(
            functionSet: functionSet,
            terminalRules: terminalRules,
            depth: depth,
            terminalProb: terminalProb
        )
        return ParseTree(root: root)
    }
    
    /**
     * Evaluates the parse tree with given variable values.
     
     This method traverses the parse tree and evaluates it based on the provided
     variable values. It returns the result of the evaluation.
     
     - Parameters:
        - variableValues: A dictionary mapping variable names to their values.
     
     - Returns: The result of evaluating the parse tree.
     */
    func evaluate(variableValues: [String: Double]) -> Double {
        return root.evaluate(variableValues: variableValues)
    }
    
    /**
     * Returns a string representation of the parse tree.
     
     This method provides a simple description of the parse tree, which is
     useful for debugging and logging purposes.
     
     - Returns: A string representation of the parse tree.
     */
    var description: String {
        return String(describing: root)
    }
    
    /**
     * Pretty prints the parse tree in a human-readable format.
     
     This method recursively traverses the parse tree and formats it as a string,
     showing the structure of the tree with indentation and branch lines.
     
     - Returns: A string representation of the parse tree.
     */
    func prettyPrint() -> String {
        func recurse(node: ParseNode, prefix: String, isTail: Bool) -> String {
            let prefix1 = isTail ? "└── " : "├── "
            var result = prefix + prefix1 + node.value + "\n"
            
            if let functionNode = node as? FunctionNode {
                let childCount = functionNode.children.count
                for i in 0..<childCount {
                    let isLast = i == childCount - 1
                    let newPrefix = prefix + (isTail ? "    " : "│   ")
                    result += recurse(node: functionNode.children[i], prefix: newPrefix, isTail: isLast)
                }
            }
            
            return result
        }
        
        return recurse(node: root, prefix: "", isTail: true)
    }
    
    /**
     * Returns all nodes in the parse tree along with their parents.
     
     This method traverses the parse tree and collects all nodes along with their
     parent nodes. It returns an array of tuples, where each tuple contains a node
     and its parent.
     
     - Returns: An array of tuples containing nodes and their parents.
     */
    private func getAllNodes() -> [(node: ParseNode, parent: ParseNode?)] {
        var nodes: [(node: ParseNode, parent: ParseNode?)] = []
        
        func traverse(node: ParseNode, parent: ParseNode?) {
            nodes.append((node, parent))
            if let functionNode = node as? FunctionNode {
                for child in functionNode.children {
                    traverse(node: child, parent: functionNode)
                }
            }
        }
        
        traverse(node: root, parent: nil)
        return nodes
    }
    
    /**
     * Returns a random node from the parse tree.
     
     This method can return a random node of any type (leaf or internal) based on the
     specified nodeType parameter. If no nodes of the specified type are found, it
     raises a fatal error.
     
     - Parameters:
        - nodeType: Type of node to return ("any", "leaf", "internal")
     
     - Returns: A tuple containing the selected node and its parent.
     */
    func getRandomNode(nodeType: String = "any") -> (node: ParseNode, parent: ParseNode?) {
        let allNodes = getAllNodes()
        var filteredNodes: [(node: ParseNode, parent: ParseNode?)] = []
        
        switch nodeType {
        case "any":
            filteredNodes = allNodes
        case "leaf":
            filteredNodes = allNodes.filter { $0.node is TerminalNode }
        case "internal":
            filteredNodes = allNodes.filter { $0.node is FunctionNode }
        default:
            filteredNodes = allNodes
        }
        
        guard !filteredNodes.isEmpty else {
            fatalError("No nodes of type \(nodeType) found in the tree")
        }
        
        let randomIndex = Int.random(in: 0..<filteredNodes.count)
        return filteredNodes[randomIndex]
    }
    
    /**
     * Creates a deep copy of the parse tree.
     
     This method creates a new instance of the parse tree with the same structure
     and values as the original tree.
     
     - Returns: A new ParseTree instance that is a copy of the original.
     */
    func copy() -> ParseTree {
        return ParseTree(root: root.copy() as! FunctionNode)
    }
    
    /**
     * Mutate the parse tree by replacing a random subtree with a new one.
     
     This method replaces a randomly selected subtree in the parse tree with
     a new subtree generated from the provided function set and terminal rules.
     
     - Parameters:
        - functionSet: Array of function names to be used in the new subtree.
        - terminalRules: The rules for generating new terminal nodes.
        - maxDepth: Maximum depth for the new subtree.
        - terminalProb: Probability of generating a terminal node.
     
     - Returns: A new parse tree with the mutated subtree.
     */
    func mutate(
        functionSet: [String],
        terminalRules: TerminalGenerationRules,
        maxDepth: Int,
        terminalProb: Double
    ) -> ParseTree {
        let mutatedTree = self.copy()
        let (nodeToReplace, parent) = mutatedTree.getRandomNode()
        let newSubtree = FunctionNode.fromFunctionSet(
            functionSet: functionSet,
            terminalRules: terminalRules,
            depth: max(1, Int.random(in: 1...maxDepth)),
            terminalProb: terminalProb
        )
        
        if parent == nil {
            mutatedTree.root = newSubtree
        } else if let parent = parent as? FunctionNode {
            for i in 0..<parent.children.count {
                if parent.children[i] === nodeToReplace {
                    parent.children[i] = newSubtree
                    break
                }
            }
        }
        
        return mutatedTree
    }
    
    /**
     * Point mutation: Replace a random terminal node with a new terminal node.
     
     This method replaces a randomly selected terminal node in the parse tree with
     a new terminal node generated from the provided terminal rules.
     
     - Parameters:
        - terminalRules: The rules for generating new terminal nodes.
     
     - Returns: A new parse tree with the mutated terminal node.
     */
    func pointMutate(terminalRules: TerminalGenerationRules) -> ParseTree {
        let mutatedTree = self.copy()
        let (nodeToReplace, parent) = mutatedTree.getRandomNode(nodeType: "leaf")
        let newTerminal = TerminalNode.fromTerminalSet(rules: terminalRules)
        
        if parent == nil {
            // This should only happen if the tree is a single node
            mutatedTree.root = FunctionNode(value: "+", children: [newTerminal, TerminalNode(value: "0")])
        } else if let parent = parent as? FunctionNode {
            for i in 0..<parent.children.count {
                if parent.children[i] === nodeToReplace {
                    parent.children[i] = newTerminal
                    break
                }
            }
        }
        
        return mutatedTree
    }
    
    /**
     * Crossover between two parse trees.
     
     This method performs crossover between two parent parse trees, creating a child tree
     by replacing a random subtree in the first parent with a random subtree from the second parent.
     
     - Parameters:
        - parent1: The first parent parse tree.
        - parent2: The second parent parse tree.
     
     - Returns: A new parse tree representing the child created from the crossover.
     */
    static func crossover(parent1: ParseTree, parent2: ParseTree) -> ParseTree {
        let child = parent1.copy()
        let (node1, parent1Node) = child.getRandomNode()
        let (node2, _) = parent2.getRandomNode()
        
        let newSubtree = node2.copy()
        
        if parent1Node == nil {
            // If the selected node is the root
            if let functionNode = newSubtree as? FunctionNode {
                child.root = functionNode
            } else {
                // If newSubtree is a terminal, we need to create a function node
                child.root = FunctionNode(value: "+", children: [newSubtree, TerminalNode(value: "0")])
            }
        } else if let parent = parent1Node as? FunctionNode {
            for i in 0..<parent.children.count {
                if parent.children[i] === node1 {
                    parent.children[i] = newSubtree
                    break
                }
            }
        }
        
        return child
    }
}

// MARK: - Genetic Programming

/**
 * Genetic Programming class for evolving parse trees.
 *
 * This class implements the genetic programming algorithm to evolve parse trees
 * for classification tasks. It includes methods for initialization, fitness evaluation,
 * selection, crossover, mutation, and multi-class classification.
 */
class GeneticProgramming {
    let functionSet: [String]
    let terminalRules: TerminalGenerationRules
    let populationSize: Int
    let maxGenerations: Int
    let maxDepth: Int
    let tournamentSize: Int
    let crossoverRate: Double
    let mutationRate: Double
    
    var population: [ParseTree] = []
    var bestFitness: Double = 0.0
    var bestIndividual: ParseTree?
    var fitnessHistory: [Double] = []
    
    /**
     * Initializes the genetic programming parameters.
     
     - Parameters:
        - functionSet: Array of function names to be used in the parse tree.
        - terminalRules: Rules for generating terminal nodes.
        - populationSize: Size of the population (default is 100).
        - maxGenerations: Maximum number of generations to evolve (default is 50).
        - maxDepth: Maximum depth of the parse trees (default is 6).
        - tournamentSize: Size of the tournament for selection (default is 7).
        - crossoverRate: Probability of crossover between parents (default is 0.9).
        - mutationRate: Probability of mutation (default is 0.1).
     */
    init(
        functionSet: [String],
        terminalRules: TerminalGenerationRules,
        populationSize: Int = 100,
        maxGenerations: Int = 50,
        maxDepth: Int = 6,
        tournamentSize: Int = 7,
        crossoverRate: Double = 0.9,
        mutationRate: Double = 0.1
    ) {
        self.functionSet = functionSet
        self.terminalRules = terminalRules
        self.populationSize = populationSize
        self.maxGenerations = maxGenerations
        self.maxDepth = maxDepth
        self.tournamentSize = tournamentSize
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
    }
    
    /**
     * Initializes the population of parse trees.
     
     This method creates a population of parse trees using a mix of full and grow methods.
     Half of the population is generated using the full method, while the other half uses
     the grow method. The depth of the trees is randomly chosen within the specified range.
     */
    func initializePopulation() {
        population = []
        
        // Create full trees for half the population
        for _ in 0..<populationSize/2 {
            let depth = Int.random(in: 2...maxDepth)
            population.append(ParseTree.generateFull(
                functionSet: functionSet,
                terminalRules: terminalRules, 
                depth: depth
            ))
        }
        
        // Create grow trees for the other half
        for _ in 0..<(populationSize - populationSize/2) {
            let depth = Int.random(in: 2...maxDepth)
            population.append(ParseTree.generateGrow(
                functionSet: functionSet,
                terminalRules: terminalRules,
                depth: depth,
                terminalProb: 0.5
            ))
        }
    }
    
    /**
     * Fitness function to evaluate the performance of a parse tree.
     
     This function calculates the fitness of a parse tree by evaluating it against
     a dataset and counting how many predictions match the actual labels.
     
     - Parameters:
        - individual: The parse tree to evaluate.
        - data: A 2D array where each inner array represents a training instance with its features.
        - labels: An array of integers representing the class labels for the training data.
        - targetClass: The class index to evolve for (default is -1 for multi-class).
     
     - Returns: A Double value representing the fitness score (accuracy).
     */
    func fitnessFunction(individual: ParseTree, data: [[Double]], labels: [Int], targetClass: Int = -1) -> Double {
        var correctCount = 0
        
        for i in 0..<data.count {
            let features = data[i]
            let actualLabel = labels[i]
            
            // Create variable values mapping
            var variableValues: [String: Double] = [:]
            for j in 0..<features.count {
                variableValues["X\(j+1)"] = features[j]
            }
            
            // Evaluate the tree to get the predicted value
            let predictedValue = individual.evaluate(variableValues: variableValues)
            
            // Binary classification: If targetClass specified, do one-vs-all
            if targetClass >= 0 {
                let predictedClass = predictedValue > 0 ? targetClass : (targetClass == 0 ? 1 : 0)
                if predictedClass == actualLabel {
                    correctCount += 1
                }
            } else {
                // Simple classification based on value range
                let predictedClass: Int
                if predictedValue < 0.33 {
                    predictedClass = 0  // Setosa
                } else if predictedValue < 0.66 {
                    predictedClass = 1  // Versicolor
                } else {
                    predictedClass = 2  // Virginica
                }
                
                if predictedClass == actualLabel {
                    correctCount += 1
                }
            }
        }
        
        return Double(correctCount) / Double(data.count)
    }
    
    /**
     * Tournament selection for selecting parents based on fitness.
     
     This method selects a parse tree from the population using tournament selection,
     which randomly selects a subset of individuals and chooses the best one among them.
     
     - Parameters:
        - fitnesses: An array of fitness values for the current population.
     
     - Returns: The selected parse tree.
     */
    func tournamentSelection(fitnesses: [Double]) -> ParseTree {
        var bestIndex = Int.random(in: 0..<populationSize)
        var bestFitness = fitnesses[bestIndex]
        
        for _ in 1..<tournamentSize {
            let index = Int.random(in: 0..<populationSize)
            let fitness = fitnesses[index]
            
            if fitness > bestFitness {
                bestIndex = index
                bestFitness = fitness
            }
        }
        
        return population[bestIndex]
    }
    
    /**
     * Evolves the population of parse trees using genetic programming.
     
     This method performs the main loop of the genetic programming algorithm,
     evolving a population of parse trees over a specified number of generations.
     
     - Parameters:
        - data: A 2D array where each inner array represents a training instance with its features.
        - labels: An array of integers representing the class labels for the training data.
        - targetClass: The class index to evolve for (default is -1 for multi-class).
     
     - Returns: The best parse tree found during evolution.
     */
    func evolve(data: [[Double]], labels: [Int], targetClass: Int = -1) -> ParseTree {
        initializePopulation()
        
        for generation in 0..<maxGenerations {
            // Calculate fitness for each individual
            let fitnesses = population.map { fitnessFunction(individual: $0, data: data, labels: labels, targetClass: targetClass) }
            
            // Find the best individual
            if let maxIndex = fitnesses.indices.max(by: { fitnesses[$0] < fitnesses[$1] }) {
                let best = fitnesses[maxIndex]
                
                // Update best individual if improved
                if best > bestFitness {
                    bestFitness = best
                    bestIndividual = population[maxIndex].copy()
                }
                
                // Track average fitness
                let avgFitness = fitnesses.reduce(0, +) / Double(fitnesses.count)
                fitnessHistory.append(avgFitness)
                
                print("Generation \(generation): Best fitness = \(bestFitness), Avg fitness = \(avgFitness)")
                
                // Return if perfect solution found
                if bestFitness >= 1.0 {
                    return population[maxIndex]
                }
            }
            
            // Create a new population
            var newPopulation: [ParseTree] = []
            
            // Elitism - keep the best individual
            if let maxIndex = fitnesses.indices.max(by: { fitnesses[$0] < fitnesses[$1] }) {
                newPopulation.append(population[maxIndex].copy())
            }
            
            // Fill the rest of the population through selection, crossover, and mutation
            while newPopulation.count < populationSize {
                // Select parents
                let parent1 = tournamentSelection(fitnesses: fitnesses)
                
                if Double.random(in: 0...1) < crossoverRate {
                    // Crossover
                    let parent2 = tournamentSelection(fitnesses: fitnesses)
                    let child = ParseTree.crossover(parent1: parent1, parent2: parent2)
                    
                    // Mutation
                    if Double.random(in: 0...1) < mutationRate {
                        newPopulation.append(child.mutate(
                            functionSet: functionSet,
                            terminalRules: terminalRules,
                            maxDepth: maxDepth,
                            terminalProb: 0.1
                        ))
                    } else {
                        newPopulation.append(child)
                    }
                } else {
                    // No crossover
                    if Double.random(in: 0...1) < mutationRate {
                        // Mutation
                        newPopulation.append(parent1.mutate(
                            functionSet: functionSet,
                            terminalRules: terminalRules,
                            maxDepth: maxDepth,
                            terminalProb: 0.1
                        ))
                    } else {
                        // Direct copy
                        newPopulation.append(parent1.copy())
                    }
                }
            }
            
            // Ensure we don't exceed population size
            while newPopulation.count > populationSize {
                newPopulation.removeLast()
            }
            
            population = newPopulation
        }
        
        // Return the best individual found during evolution
        return bestIndividual ?? population[0]
    }
    
    /**
     * Solves the multi-class classification problem using genetic programming.
     
     This method trains multiple classifiers, one for each class, using the provided
     training data and labels. It returns an array of parse trees representing the
     trained classifiers.
     
     - Parameters:
        - data: A 2D array where each inner array represents a training instance with its features.
        - labels: An array of integers representing the class labels for the training data.
        - numClasses: The number of classes in the dataset (default is 3 for Iris dataset).
     
     - Returns: An array of parse trees representing the trained classifiers for each class.
     */
    func solveMultiClass(data: [[Double]], labels: [Int], numClasses: Int = 3) -> [ParseTree] {
        var classifiers: [ParseTree] = []
        
        // Train one classifier per class
        for classIndex in 0..<numClasses {
            print("\nTraining classifier for class \(classIndex)")
            let classifier = evolve(data: data, labels: labels, targetClass: classIndex)
            classifiers.append(classifier)
            print("Best fitness for class \(classIndex): \(bestFitness)")
            
            // Reset for next class
            bestFitness = 0.0
            bestIndividual = nil
        }
        
        return classifiers
    }
    
    /**
     Tests the performance of multiple classifiers on the provided test data.
     
     This method evaluates how well the given classifiers perform on the test dataset
     by comparing their predictions against the provided test labels.
     
     - Parameters:
        - classifiers: An array of parse trees representing different classifiers.
        - testData: A 2D array where each inner array represents a test instance with its features.
        - testLabels: The correct class labels for the test data.
     
     - Returns: A performance metric (likely accuracy) as a Double value, indicating how well the
        classifiers performed on the test data.
     */
    func testMultiClass(classifiers: [ParseTree], testData: [[Double]], testLabels: [Int]) -> Double {
        var correctCount = 0
        
        for i in 0..<testData.count {
            let features = testData[i]
            let actualLabel = testLabels[i]
            
            // Create variable values mapping
            var variableValues: [String: Double] = [:]
            for j in 0..<features.count {
                variableValues["X\(j+1)"] = features[j]
            }
            
            // Get confidence scores from each classifier
            var confidences: [Double] = []
            for classifier in classifiers {
                confidences.append(classifier.evaluate(variableValues: variableValues))
            }
            
            // Predict class with highest confidence
            if let maxIndex = confidences.indices.max(by: { confidences[$0] < confidences[$1] }) {
                if maxIndex == actualLabel {
                    correctCount += 1
                }
            }
        }
        
        let accuracy = Double(correctCount) / Double(testData.count)
        print("\nTest accuracy: \(accuracy * 100)%")
        return accuracy
    }
}

// MARK: - Benchmarking Extension

/**
 * Extension for GeneticProgramming to add benchmarking capabilities.
 *
 * This extension provides methods to benchmark the performance of the genetic programming algorithm
 * with different population sizes and generation counts. It allows for easy evaluation of how these
 * parameters affect the execution time and accuracy of the algorithm.
 */
extension GeneticProgramming {
    
    /**
     Benchmarks the genetic programming algorithm's performance with various population sizes.
     
     This function evaluates how different population sizes affect the performance and accuracy
     of the genetic programming algorithm. For each specified population size, it creates a new GP instance,
     measures the time taken to train classifiers on the training data, and evaluates the accuracy
     on the test data.
     
     - Parameters:
        - sizes: An array of integer values representing different population sizes to benchmark
        - trainData: A 2D array of Doubles containing the training data features
        - trainLabels: An array of integers representing the class labels for the training data
        - testData: A 2D array of Doubles containing the test data features
        - testLabels: An array of integers representing the class labels for the test data
        
     - Returns: An array of tuples, where each tuple contains:
        - size: The population size used
        - time: The execution time in seconds
        - accuracy: The classification accuracy achieved on the test data
     
     - Note: All other genetic programming parameters (like generation count, mutation rate, etc.)
                are taken from the current instance's properties.
     */
    func runPopulationSizeBenchmark(sizes: [Int], trainData: [[Double]], trainLabels: [Int], testData: [[Double]], testLabels: [Int]) -> [(size: Int, time: Double, accuracy: Double)] {
        var results: [(size: Int, time: Double, accuracy: Double)] = []
        
        for size in sizes {
            print("\nRunning benchmark with population size: \(size)")
            
            // Create new GP instance with this population size
            let gp = GeneticProgramming(
                functionSet: self.functionSet,
                terminalRules: self.terminalRules,
                populationSize: size,
                maxGenerations: self.maxGenerations,
                maxDepth: self.maxDepth,
                tournamentSize: self.tournamentSize,
                crossoverRate: self.crossoverRate,
                mutationRate: self.mutationRate
            )
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let classifiers = gp.solveMultiClass(data: trainData, labels: trainLabels)
            let endTime = CFAbsoluteTimeGetCurrent()
            let executionTime = endTime - startTime
            
            let accuracy = gp.testMultiClass(classifiers: classifiers, testData: testData, testLabels: testLabels)
            
            results.append((size: size, time: executionTime, accuracy: accuracy))
            print("Population size \(size): \(executionTime) seconds, Accuracy: \(accuracy)")
        }
        
        return results
    }

    /**
     Benchmarks the genetic programming algorithm's performance with various generation count settings.
     
     This function evaluates how different maximum generation counts affect the performance and accuracy
     of the genetic programming algorithm. For each specified generation count, it creates a new GP instance,
     measures the time taken to train classifiers on the training data, and evaluates the accuracy
     on the test data.
     
     - Parameters:
        - counts: An array of integer values representing different maximum generation counts to benchmark
        - trainData: A 2D array of Doubles containing the training data features
        - trainLabels: An array of integers representing the class labels for the training data
        - testData: A 2D array of Doubles containing the test data features
        - testLabels: An array of integers representing the class labels for the test data
        
     - Returns: An array of tuples, where each tuple contains:
        - count: The generation count used
        - time: The execution time in seconds
        - accuracy: The classification accuracy achieved on the test data
     
     - Note: All other genetic programming parameters (like population size, mutation rate, etc.)
                are taken from the current instance's properties.
     */
    func runGenerationCountBenchmark(counts: [Int], trainData: [[Double]], trainLabels: [Int], testData: [[Double]], testLabels: [Int]) -> [(count: Int, time: Double, accuracy: Double)] {
        var results: [(count: Int, time: Double, accuracy: Double)] = []
        
        for count in counts {
            print("\nRunning benchmark with generation count: \(count)")
            
            // Create new GP instance with this generation count
            let gp = GeneticProgramming(
                functionSet: self.functionSet,
                terminalRules: self.terminalRules,
                populationSize: self.populationSize,
                maxGenerations: count,
                maxDepth: self.maxDepth,
                tournamentSize: self.tournamentSize,
                crossoverRate: self.crossoverRate,
                mutationRate: self.mutationRate
            )
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let classifiers = gp.solveMultiClass(data: trainData, labels: trainLabels)
            let endTime = CFAbsoluteTimeGetCurrent()
            let executionTime = endTime - startTime
            
            let accuracy = gp.testMultiClass(classifiers: classifiers, testData: testData, testLabels: testLabels)
            
            results.append((count: count, time: executionTime, accuracy: accuracy))
            print("Generation count \(count): \(executionTime) seconds, Accuracy: \(accuracy)")
        }
        
        return results
    }
}

/// A structure to hold benchmark results.
struct BenchmarkResult: Codable {
    /// The parameter name (e.g., "Population Size", "Generations").
    let parameter: String

    /// The value of the parameter (e.g., population size or generation count).
    let value: Int
    
    /// The execution time in seconds.
    let time: Double
    
    /// The accuracy achieved during the benchmark.
    let accuracy: Double
}

// MARK: - Utility Functions

/**
 Saves benchmark results to a CSV file.
 
 This function takes an array of benchmark results and saves them to a specified CSV file.
 The CSV file will contain the parameter name, value, execution time, and accuracy for each result.
 
 - Parameters:
    - results: An array of `BenchmarkResult` objects containing the benchmark data.
    - filename: The name of the CSV file to save the results to.
 */
func saveResultsToCSV(results: [BenchmarkResult], filename: String) {
    // Create CSV content
    var csvContent = "parameter,value,time,accuracy\n"
    
    for result in results {
        csvContent += "\(result.parameter),\(result.value),\(result.time),\(result.accuracy)\n"
    }
    
    // Save to file
    do {
        try csvContent.write(toFile: filename, atomically: true, encoding: .utf8)
        print("Results saved to \(filename)")
    } catch {
        print("Error saving results: \(error.localizedDescription)")
    }
}

/**
 Loads the Iris dataset from a CSV file.
 
 This function attempts to load the Iris dataset from a specified CSV file.
 If the file is not found in the main bundle, it falls back to a default path.
 
 - Returns: An instance of `IrisDataset` containing the loaded data.
 */
func loadIrisDataset() -> IrisDataset {
    let dataset = IrisDataset()
    
    do {
        // Try to load from bundle first
        if let fileURL = Bundle.main.url(forResource: "Iris", withExtension: "csv") {
            try dataset.loadFromCSV(filePath: fileURL.path)
        } else {
            // Fallback to direct path
            let filePath = "../data/Iris.csv" // Adjust path as needed
            try dataset.loadFromCSV(filePath: filePath)
        }
    } catch {
        print("Error loading dataset: \(error.localizedDescription)")
    }
    
    return dataset
}

// MARK: - Main Benchmark Function

/**
 Runs a suite of benchmarks for genetic programming in Swift.
 
 This function benchmarks the performance of genetic programming
 algorithms on the Iris dataset. It evaluates the impact of different
 population sizes and generation counts on execution time and accuracy.
 Call this function to execute the benchmarks and save results to a CSV file.
 
 - Note: Ensure that the Iris dataset is available at the specified path.
 
 - Parameters:
    - None
 - Returns:
    - None
 */
func runSwiftBenchmarks() {
    print("Running Swift GP benchmarks...\n")
    
    // Load dataset
    let dataset = loadIrisDataset()
    let (trainData, testData) = dataset.splitData(trainRatio: 0.7)
    
    // Prepare data
    let trainFeatures = trainData.map { sample in 
        [sample.features.sepalLength, sample.features.sepalWidth, 
         sample.features.petalLength, sample.features.petalWidth] 
    }
    let trainLabels = trainData.map { $0.speciesIndex }
    
    let testFeatures = testData.map { sample in 
        [sample.features.sepalLength, sample.features.sepalWidth, 
         sample.features.petalLength, sample.features.petalWidth] 
    }
    let testLabels = testData.map { $0.speciesIndex }
    
    // Configure benchmark parameters
    let populationSizes = [50, 100, 200]
    let generationCounts = [10, 20, 50]
    
    // Create base GP instance
    let functionSet = ["+", "-", "*", "/"]
    let terminalRules = TerminalGenerationRules(
        literals: dataset.getFeatureNames(),
        constantsRange: (-10, 10)
    )
    
    let gp = GeneticProgramming(
        functionSet: functionSet,
        terminalRules: terminalRules,
        populationSize: 100,
        maxGenerations: 20,
        maxDepth: 4,
        tournamentSize: 7,
        crossoverRate: 0.9,
        mutationRate: 0.1
    )
    
    // Run population size benchmarks
    print("\nBenchmarking with varying population size...")
    let popResults = gp.runPopulationSizeBenchmark(
        sizes: populationSizes,
        trainData: trainFeatures, 
        trainLabels: trainLabels,
        testData: testFeatures,
        testLabels: testLabels
    )
    
    // Run generation count benchmarks
    print("\nBenchmarking with varying generation count...")
    let genResults = gp.runGenerationCountBenchmark(
        counts: generationCounts,
        trainData: trainFeatures, 
        trainLabels: trainLabels,
        testData: testFeatures,
        testLabels: testLabels
    )
    
    // Prepare results for saving
    var benchmarkResults: [BenchmarkResult] = []
    
    for (size, time, accuracy) in popResults {
        benchmarkResults.append(BenchmarkResult(
            parameter: "Population Size",
            value: size,
            time: time,
            accuracy: accuracy
        ))
    }
    
    for (count, time, accuracy) in genResults {
        benchmarkResults.append(BenchmarkResult(
            parameter: "Generations",
            value: count,
            time: time,
            accuracy: accuracy
        ))
    }
    
    // Save results
    saveResultsToCSV(results: benchmarkResults, filename: "../data/swift_gp_benchmarks.csv")
    print("\nBenchmark results saved to '../data/swift_gp_benchmarks.csv'")
}

// MARK: - Main Execution

print("\n===== RUNNING BENCHMARKS =====")
runSwiftBenchmarks()
