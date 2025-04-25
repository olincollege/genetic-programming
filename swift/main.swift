import Foundation

// MARK: - Iris Data Structures

struct IrisFeatures {
    var sepalLength: Double
    var sepalWidth: Double
    var petalLength: Double
    var petalWidth: Double
}

struct IrisSample {
    let features: IrisFeatures
    let species: String
    let speciesIndex: Int // 0 = setosa, 1 = versicolor, 2 = virginica
}

// MARK: - Iris Dataset Manager

class IrisDataset {
    private(set) var samples: [IrisSample] = []
    private(set) var normalizedSamples: [IrisSample] = []
    
    // Maps for species to index conversion
    private let speciesMap = [
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    ]
    
    // Min and max values for each feature (used for normalization)
    private var minValues = IrisFeatures(sepalLength: Double.infinity, sepalWidth: Double.infinity, 
                                         petalLength: Double.infinity, petalWidth: Double.infinity)
    private var maxValues = IrisFeatures(sepalLength: -Double.infinity, sepalWidth: -Double.infinity, 
                                         petalLength: -Double.infinity, petalWidth: -Double.infinity)
    
    // Load data from a CSV file
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
    
    // Update min and max values for normalization
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
    
    // Normalize data to [0,1] range
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
    
    // Helper to normalize a single value
    private func normalize(_ value: Double, min: Double, max: Double) -> Double {
        return (value - min) / (max - min)
    }
    
    // Split data into training and testing sets
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
    
    // Convert data to array format for genetic programming
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
    
    // Get feature names for terminal set
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
        
        // Placeholder - actual implementation needed
        return self
    }
    
    // Crossover: Swap subtrees between two parse trees
    static func crossover(
        parent1: ParseTree,
        parent2: ParseTree
    ) -> (ParseTree, ParseTree) {
        // Create deep copies of both parents
        // Select random crossover points in each parent
        // Swap the subtrees at these points
        
        // Placeholder - actual implementation needed
        return (parent1, parent2)
    }
}

// Helper extension to enable description of parse trees
extension FunctionNode: CustomStringConvertible {
    var description: String {
        let args = children.map { String(describing: $0) }.joined(separator: " ")
        return "(\(value) \(args))"
    }
}

extension TerminalNode: CustomStringConvertible {
    var description: String {
        return value
    }
}

extension ParseTree: CustomStringConvertible {
    var description: String {
        return String(describing: root)
    }
}

// Example implementation for genetic programming with the Iris dataset
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
            let depth = Int.random(in: 1...maxDepth)
            population.append(ParseTree.generateFull(
                functionSet: functionSet,
                terminalRules: terminalRules, 
                depth: depth
            ))
        }
        
        // Create grow trees for the other half
        for _ in 0..<(populationSize - populationSize/2) {
            let depth = Int.random(in: 1...maxDepth)
            population.append(ParseTree.generateGrow(
                functionSet: functionSet,
                terminalRules: terminalRules,
                depth: depth,
                terminalProb: 0.5
            ))
        }
    }
    
    // Fitness function for Iris classification
    func fitnessFunction(individual: ParseTree, data: [[Double]], labels: [Int]) -> Double {
        // Placeholder fitness function
        // Should evaluate how well the tree classifies the Iris data
        return 0.0
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
    func evolve(data: [[Double]], labels: [Int]) -> ParseTree {
        initializePopulation()
        
        for generation in 0..<maxGenerations {
            // Calculate fitness for each individual
            let fitnesses = population.map { fitnessFunction(individual: $0, data: data, labels: labels) }
            
            // Find the best individual
            if let maxIndex = fitnesses.indices.max(by: { fitnesses[$0] < fitnesses[$1] }) {
                let bestFitness = fitnesses[maxIndex]
                print("Generation \(generation): Best fitness = \(bestFitness)")
                
                // Return if perfect solution found
                if bestFitness >= 1.0 {
                    return population[maxIndex]
                }
            }
            
            // Create a new population
            var newPopulation: [ParseTree] = []
            
            // Elitism - keep the best individual
            if let maxIndex = fitnesses.indices.max(by: { fitnesses[$0] < fitnesses[$1] }) {
                newPopulation.append(population[maxIndex])
            }
            
            // Fill the rest of the population through selection, crossover, and mutation
            while newPopulation.count < populationSize {
                let parent1 = tournamentSelection(fitnesses: fitnesses)
                
                // Crossover with probability crossoverRate
                if Double.random(in: 0...1) < crossoverRate {
                    let parent2 = tournamentSelection(fitnesses: fitnesses)
                    let (child1, child2) = ParseTree.crossover(parent1: parent1, parent2: parent2)
                    
                    // Mutation with probability mutationRate
                    if Double.random(in: 0...1) < mutationRate {
                        newPopulation.append(child1.mutate(
                            functionSet: functionSet,
                            terminalRules: terminalRules,
                            maxDepth: maxDepth,
                            terminalProb: 0.5
                        ))
                    } else {
                        newPopulation.append(child1)
                    }
                    
                    if newPopulation.count < populationSize {
                        if Double.random(in: 0...1) < mutationRate {
                            newPopulation.append(child2.mutate(
                                functionSet: functionSet,
                                terminalRules: terminalRules,
                                maxDepth: maxDepth,
                                terminalProb: 0.5
                            ))
                        } else {
                            newPopulation.append(child2)
                        }
                    }
                } else {
                    // No crossover, just copy
                    if Double.random(in: 0...1) < mutationRate {
                        newPopulation.append(parent1.mutate(
                            functionSet: functionSet,
                            terminalRules: terminalRules,
                            maxDepth: maxDepth,
                            terminalProb: 0.5
                        ))
                    } else {
                        newPopulation.append(parent1)
                    }
                }
            }
            
            // Make sure we don't exceed population size
            while newPopulation.count > populationSize {
                newPopulation.removeLast()
            }
            
            population = newPopulation
        }
        
        // Return the best individual from the final population
        let finalFitnesses = population.map { fitnessFunction(individual: $0, data: data, labels: labels) }
        let bestIndex = finalFitnesses.indices.max(by: { finalFitnesses[$0] < finalFitnesses[$1] }) ?? 0
        
        return population[bestIndex]
    }
}

// MARK: - Fitness Function for Iris Classification

extension GeneticProgramming {
    // Binary classification fitness function (one species vs others)
    func binaryFitness(individual: ParseTree, data: [[Double]], labels: [Int], targetClass: Int) -> Double {
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
            
            // For binary classification:
            // If predictedValue > 0, predict targetClass, else predict other class
            let predictedClass = predictedValue > 0 ? targetClass : (targetClass == 0 ? 1 : 0)
            
            if predictedClass == actualLabel {
                correctCount += 1
            }
        }
        
        return Double(correctCount) / Double(data.count)
    }
    
    // Multi-class classification using "one-vs-all" approach with multiple trees
    func multiClassFitness(individuals: [ParseTree], data: [[Double]], labels: [Int]) -> Double {
        var correctCount = 0
        
        for i in 0..<data.count {
            let features = data[i]
            let actualLabel = labels[i]
            
            // Create variable values mapping
            var variableValues: [String: Double] = [:]
            for j in 0..<features.count {
                variableValues["X\(j+1)"] = features[j]
            }
            
            // Evaluate each tree to get confidence for each class
            var confidences: [Double] = []
            for tree in individuals {
                confidences.append(tree.evaluate(variableValues: variableValues))
            }
            
            // Predict the class with highest confidence
            let predictedClass = confidences.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            
            if predictedClass == actualLabel {
                correctCount += 1
            }
        }
        
        return Double(correctCount) / Double(data.count)
    }
}

// MARK: - Main Execution

// 1. Load the dataset
let dataset = loadIrisDataset()
let (features, labels) = dataset.getDataArrays()

// 2. Setup genetic programming parameters
let functionSet = ["+", "-", "*", "/"] // Could add "sin", "cos", etc.
let terminalRules = TerminalGenerationRules(
    literals: dataset.getFeatureNames(),
    constantsRange: (-10, 10),
    intsOnly: false,
    noRandomConstants: false
)

// 3. Create and run genetic programming algorithm
let gp = GeneticProgramming(
    functionSet: functionSet,
    terminalRules: terminalRules,
    populationSize: 100,
    maxGenerations: 50
)

// 4. Implement missing methods:
// - Complete the mutate method in ParseTree
// - Complete the crossover method in ParseTree
// - Implement the fitnessFunction inside GeneticProgramming

// 5. Run the genetic programming algorithm
let bestSolution = gp.evolve(data: features, labels: labels)

// 6. Test the best solution
print("Best solution: \(bestSolution)")
print("Testing solution...")

// Create test data split
let (trainData, testData) = dataset.splitData(trainRatio: 0.7)
