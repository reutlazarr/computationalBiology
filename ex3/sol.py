import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  # Import the time module

# Load the data
data_path = './digits_test.csv'
keys_path = './digits_keys.csv'
data = pd.read_csv(data_path, header=None).values
digits_keys = pd.read_csv(keys_path, header=None).values.flatten()

# Parameters
num_neurons = 10  # Number of neurons in each dimension (10x10 grid)
input_len = 784  # Length of input vector (28x28 images)
radius = 1.0  # Neighborhood radius
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

def plot_som_with_labels(weights, data, labels):
    plt.figure(figsize=(10, 10))
    label_map = np.zeros((num_neurons, num_neurons, 10))  # To count the labels in each neuron
    for x, label in zip(data, labels):
        dists = euclidean_distance(weights, x)
        bmu_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        label_map[bmu_idx[0], bmu_idx[1], int(label)] += 1
    
    for i in range(num_neurons):
        for j in range(num_neurons):
            if np.sum(label_map[i, j, :]) > 0:
                label = np.argmax(label_map[i, j, :])
                percentage = label_map[i, j, label] / np.sum(label_map[i, j, :]) * 100
                plt.text(i + .5, j + .5, f'{label}\n{percentage:.1f}%', 
                         color='black', fontdict={'weight': 'bold', 'size': 12},
                         ha='center', va='center')
    plt.xlim([0, num_neurons])
    plt.ylim([0, num_neurons])
    plt.grid()
    plt.title("SOM Visualization with Dominant Digit and Percentage")
    plt.gca().invert_yaxis()
    plt.show()


def plot_som_neurons(weights):
    plt.figure(figsize=(10, 10))
    for i in range(num_neurons):
        for j in range(num_neurons):
            plt.subplot(num_neurons, num_neurons, i * num_neurons + j + 1)
            plt.imshow(weights[i, j].reshape(28, 28), cmap='gray')
            plt.axis('off')
    plt.suptitle("SOM Neuron Weights Visualization")
    plt.show()

# Plot the result
plot_som_with_labels(weights, data, digits_keys)

# Plot the neuron weights as images
plot_som_neurons(weights)
