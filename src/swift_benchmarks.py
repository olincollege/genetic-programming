import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_swift_benchmarks():
    """Generate visualization graphs for Swift genetic programming benchmarks"""
    # Load Swift benchmark data
    swift_data = pd.read_csv('../data/swift_gp_benchmarks.csv')
    
    # Ensure the output directory exists
    os.makedirs('../docs/img', exist_ok=True)
    
    try:
        # Plot runtime results
        plt.figure(figsize=(12, 6))
        
        # Plot population size vs time benchmarks
        pop_results = swift_data[swift_data['parameter'] == 'Population Size']
        plt.subplot(1, 2, 1)
        plt.plot(pop_results['value'], pop_results['time'], 'o-', color='#F05138', label='Execution Time')
        plt.xlabel('Population Size')
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time vs Population Size')
        plt.grid(True)
        
        # Plot generation vs time benchmarks
        gen_results = swift_data[swift_data['parameter'] == 'Generations']
        plt.subplot(1, 2, 2)
        plt.plot(gen_results['value'], gen_results['time'], 'o-', color='#F05138', label='Execution Time')
        plt.xlabel('Generations')
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time vs Generations')
        plt.grid(True)
        
        plt.suptitle('Swift Genetic Programming: Runtime Performance', fontsize=14)
        plt.tight_layout()
        plt.savefig('../docs/img/swift_benchmarks.png')
        plt.close()
        
        # Plot accuracy results (as a bonus)
        plt.figure(figsize=(12, 6))
        
        # Plot population size vs accuracy benchmarks
        plt.subplot(1, 2, 1)
        plt.plot(pop_results['value'], pop_results['accuracy'], 'o-', color='#F05138', label='Accuracy')
        plt.xlabel('Population Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Population Size')
        plt.grid(True)
        
        # Plot generation vs accuracy benchmarks
        plt.subplot(1, 2, 2)
        plt.plot(gen_results['value'], gen_results['accuracy'], 'o-', color='#F05138', label='Accuracy')
        plt.xlabel('Generations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Generations')
        plt.grid(True)
        
        plt.suptitle('Swift Genetic Programming: Classification Accuracy', fontsize=14)
        plt.tight_layout()
        plt.savefig('../docs/img/swift_accuracy.png')
        
        print("\nSwift visualizations generated successfully:")
        print("- ../docs/img/swift_benchmarks.png (Runtime Performance)")
        print("- ../docs/img/swift_accuracy.png (Classification Accuracy)")
        
    except Exception as e:
        print(f"Error creating plots: {str(e)}")

if __name__ == "__main__":
    visualize_swift_benchmarks()
