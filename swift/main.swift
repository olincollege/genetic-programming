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
     *
     * @param filePath Path to the CSV file containing Iris data
     * @throws Error if file cannot be read or parsed
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
     *
     * @param features An IrisFeatures instance to compare against current min/max
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
     *
     * Uses the previously calculated min/max values to perform min-max scaling.
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
     * Normalizes a single value using min-max scaling.
     *
     * @param value The value to normalize
     * @param min Minimum value for the feature
     * @param max Maximum value for the feature
     * @return Normalized value in [0,1] range
     */
    private func normalize(_ value: Double, min: Double, max: Double) -> Double {
        return min == max ? 0 : (value - min) / (max - min)
    }
    
    /**
     * Splits the normalized dataset into training and testing subsets.
     *
     * @param trainRatio Proportion of data to use for training (default: 0.7)
     * @return Tuple containing arrays of training and testing samples
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
     *
     * @return Tuple containing feature matrix and label vector
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
     * Returns feature names for constructing terminal set in genetic programming.
     *
     * @return Array of strings representing feature names (X1, X2, X3, X4)
     */
    func getFeatureNames() -> [String] {
        return ["X1", "X2", "X3", "X4"] // Corresponding to the 4 Iris features
    }
}

// Iris Dataset Loading - Usage Example

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

// MARK: - Genetic Programming Structures

/**
 * Rules for generating terminal nodes in a parse tree.
 */
class TerminalGenerationRules {
    let literals: [String]
    let constantsRange: (Double, Double)
    let decimalPlaces: Int
    let intsOnly: Bool
    let noRandomConstants: Bool
    
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
 * Protocol for nodes in a parse tree.
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
    
    init(value: String) {
        self.value = value
    }
    
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
    
    func evaluate(variableValues: [String: Double]) -> Double {
        if let variableValue = variableValues[value] {
            return variableValue
        }
        
        if let numericValue = Double(value) {
            return numericValue
        }
        
        fatalError("Invalid terminal value: \(value)")
    }
    
    var description: String {
        return value
    }
    
    func copy() -> ParseNode {
        return TerminalNode(value: self.value)
    }
}

/**
 * A function node in the parse tree, representing an operation with children nodes.
 */
class FunctionNode: ParseNode {
    let value: String
    let arity: Int
    var children: [ParseNode]
    
    init(value: String, children: [ParseNode]) {
        self.value = value
        self.arity = FunctionNode.arityMap(function: value)
        self.children = children
    }
    
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
    
    var description: String {
        let args = children.map { String(describing: $0) }.joined(separator: " ")
        return "(\(value) \(args))"
    }
    
    func copy() -> ParseNode {
        let copiedChildren = children.map { $0.copy() }
        return FunctionNode(value: self.value, children: copiedChildren)
    }
}

/**
 * A parse tree, representing a mathematical expression in a tree structure.
 */
class ParseTree: CustomStringConvertible {
    var root: FunctionNode
    
    init(root: FunctionNode) {
        self.root = root
    }
    
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
    
    func evaluate(variableValues: [String: Double]) -> Double {
        return root.evaluate(variableValues: variableValues)
    }
    
    var description: String {
        return String(describing: root)
    }
    
    // Pretty print for better visualization 
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
    
    // Helper function to get all nodes in the tree
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
    
    // Get a random node from the tree
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
    
    // Make a deep copy of the tree
    func copy() -> ParseTree {
        return ParseTree(root: root.copy() as! FunctionNode)
    }
    
    // Mutation: Replace a random subtree with a new randomly generated subtree
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
    
    // Mutation: Replace a random terminal node
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
    
    // Crossover: Swap subtrees between two parse trees
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
    
    // Initialize the population with random parse trees
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
    
    // Fitness function for Iris classification
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
    
    // Tournament selection
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
    
    // Evolve the population for a number of generations
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
    
    // Multi-class classification using "one-vs-all" approach with multiple trees
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
    
    // Test multi-class classification accuracy
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

extension GeneticProgramming {
    // Run benchmarks with different population sizes
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
    
    // Run benchmarks with different generation counts
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

// MARK: - Benchmark Results Storage

struct BenchmarkResult: Codable {
    let parameter: String
    let value: Int
    let time: Double
    let accuracy: Double
}

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

// MARK: - Main Benchmark Function

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

// Add this to the end of main.swift
print("\n===== RUNNING BENCHMARKS =====")
runSwiftBenchmarks()
