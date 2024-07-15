import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  # Import the time module

# Load the data
data_path = './digits_test.csv'
keys_path = './digits_keys.csv'
data = pd.read_csv(data_path, header=None).values
data = data / 256
digits_keys = pd.read_csv(keys_path, header=None).values.flatten()

# Parameters
num_neurons = 10  # Number of neurons in each dimension (10x10 grid)
input_len = 784  # Length of input vector (28x28 images)
radius = 2.5  # Neighborhood radius
learning_rate = 0.05  # Learning rate

# Calculate the overall mean of all training examples
num_iterations = 10  # Number of iterations
max_time = 3 * 60  # Maximum time in seconds (3 minutes)

# Calculate the overall mean of all training examples
mean_vector = np.mean(data, axis=0)

# Initialize the weights close to the mean vector with small random noise
weights = np.tile(mean_vector, (num_neurons, num_neurons, 1)) + np.random.normal(0, 0.1, (num_neurons, num_neurons, input_len))

def euclidean_distance(x, y):
    return np.linalg.norm(x - y, axis=-1)

def decay_function(initial, t, max_iter):
    return initial * (1-t/max_iter)

def train_som(data, weights, num_iterations, radius, learning_rate, max_time):
    start_time = time.time()  # Start timing
    for t in range(num_iterations):
        iteration_start_time = time.time()  # Timing for each iteration
        for vector in data:
            # Find the Best Matching Unit (BMU)
            dists = euclidean_distance(weights, vector)
            bmu_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            
            # Update the weights
            for i in range(num_neurons):
                for j in range(num_neurons):
                    neuron_pos = np.array([i, j])
                    dist_to_bmu = euclidean_distance(neuron_pos, bmu_idx)
                    if dist_to_bmu <= radius:
                        neighborhood = np.exp(-dist_to_bmu ** 2 / (2 * radius ** 2))
                        current_error = vector - weights[i, j]
                        weights[i, j] += neighborhood * learning_rate * current_error
        
        # Decay radius and learning rate
        radius = decay_function(radius, t, num_iterations)
        learning_rate = decay_function(learning_rate, t, num_iterations)
        iteration_end_time = time.time()  # End timing for each iteration
        elapsed_time = iteration_end_time - start_time  # Total elapsed time
        print(f"Iteration {t+1}/{num_iterations} took {iteration_end_time - iteration_start_time:.2f} seconds")
        
        if elapsed_time >= max_time:
            print("Stopping training due to time limit")
            break

train_som(data, weights, num_iterations, radius, learning_rate, max_time)

def quantization_error(data, weights):
    total_error = 0
    for vector in data:
        dists = euclidean_distance(weights, vector)
        min_dist = np.min(dists)
        total_error += min_dist
    return total_error / len(data)

def is_adjacent(pos1, pos2):
    return np.sum(np.abs(np.array(pos1) - np.array(pos2))) == 1

def topological_error(data, weights):
    bad_mappings = 0
    for vector in data:
        dists = euclidean_distance(weights, vector)
        bmu_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        dists[bmu_idx] = np.inf  # Exclude the BMU itself
        second_bmu_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        if not is_adjacent(bmu_idx, second_bmu_idx):
            bad_mappings += 1
    return bad_mappings / len(data)

quant_error = quantization_error(data, weights)
topo_error = topological_error(data, weights)

print(f"Quantization Error: {quant_error}")
print(f"Topological Error: {topo_error}")


def plot_som_neurons(weights):
    plt.figure(figsize=(10, 10))
    for i in range(num_neurons):
        for j in range(num_neurons):
            plt.subplot(num_neurons, num_neurons, i * num_neurons + j + 1)
            plt.imshow(weights[i, j].reshape(28, 28), cmap='gray')
            plt.axis('off')
    plt.suptitle("SOM Neuron Weights Visualization")
    plt.show()

# Plot the neuron weights as images
plot_som_neurons(weights)
