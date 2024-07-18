import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Load the data
data_path = './digits_test.csv'
keys_path = './digits_keys.csv'

data = pd.read_csv(data_path, header=None).values
# Normalize the data
data = data / 256

digits_keys = pd.read_csv(keys_path, header=None).values.flatten()

# Parameters
num_neurons = 10  # Number of neurons in each dimension (10x10 grid)
input_len = 784  # Length of input vector (28x28 images)
radius = 2  # Neighborhood radius
learning_rate = 0.15  # Learning rate

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
    return initial * np.exp(-t / max_iter)

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

# Function to find clusters
def find_clusters(weights, radius):
    clusters = np.zeros((num_neurons, num_neurons), dtype=int)
    cluster_id = 0
    visited = np.zeros((num_neurons, num_neurons), dtype=bool)

    def dfs(i, j):
        stack = [(i, j)]
        clusters[i, j] = cluster_id
        while stack:
            x, y = stack.pop()
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < num_neurons and 0 <= ny < num_neurons and not visited[nx, ny]:
                        if euclidean_distance(weights[x, y], weights[nx, ny]) < radius:
                            clusters[nx, ny] = cluster_id
                            visited[nx, ny] = True
                            stack.append((nx, ny))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if not visited[i, j]:
                visited[i, j] = True
                dfs(i, j)
                cluster_id += 1

    return clusters

# Find clusters in the SOM
clusters = find_clusters(weights, radius=0.5)

# Plot the neuron weights with cluster borders
def plot_som_neurons_with_clusters(weights, clusters):
    plt.figure(figsize=(10, 10))
    unique_clusters = np.unique(clusters)
    colors = plt.cm.get_cmap('hsv', len(unique_clusters))

    for i in range(num_neurons):
        for j in range(num_neurons):
            plt.subplot(num_neurons, num_neurons, i * num_neurons + j + 1)
            plt.imshow(weights[i, j].reshape(28, 28), cmap='gray')
            plt.axis('off')
            for cluster_id in unique_clusters:
                if clusters[i, j] == cluster_id:
                    plt.gca().add_patch(plt.Rectangle((0, 0), 27, 27, linewidth=2, edgecolor=colors(cluster_id), facecolor='none'))
                    break

    plt.suptitle("SOM Neuron Weights Visualization with Clusters")
    plt.show()

# Plot the neuron weights with cluster borders
plot_som_neurons_with_clusters(weights, clusters)
