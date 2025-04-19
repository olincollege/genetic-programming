import Foundation

// Iris Data Structures

struct IrisFeatures {
    let sepalLength: Double
    let sepalWidth: Double
    let petalLength: Double
    let petalWidth: Double
}

struct IrisSample {
    let features: IrisFeatures
    let species: String
    let speciesIndex: Int // 0 = setosa, 1 = versicolor, 2 = virginica
}

// Iris Dataset Manager

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
            let filePath = "data/Iris.csv" // Adjust path as needed
            try dataset.loadFromCSV(filePath: filePath)
        }
    } catch {
        print("Error loading dataset: \(error.localizedDescription)")
    }
    
    return dataset
}
