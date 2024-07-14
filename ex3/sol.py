import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_path = './digits_test.csv'
keys_path = './digits_keys.csv'
data = pd.read_csv(data_path, header=None).values
digits_keys = pd.read_csv(keys_path, header=None).values.flatten()

# Parameters
num_neurons = 10  # Number of neurons in each dimension (10x10 grid)
input_len = 784  # Length of input vector (28x28 images)
raduis = 1.0  # Neighborhood radius
learning_rate = 0.5  # Learning rate
num_iterations = 10000  # Number of iterations

# Initialize the weights
weights = np.random.rand(num_neurons, num_neurons, input_len) * 255

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def decay_function(initial, t, max_iter):
    return initial * np.exp(-t / max_iter)

def train_som(data, weights, num_iterations, raduis, learning_rate):
    for t in range(num_iterations):
        for vector in data:
            # Find the Best Matching Unit (BMU)
            bmu_idx = np.array([0, 0])
            min_dist = np.inf
            for i in range(num_neurons):
                for j in range(num_neurons):
                    dist = euclidean_distance(vector, weights[i, j])
                    if dist < min_dist:
                        min_dist = dist
                        bmu_idx = np.array([i, j])
            
            # Update the weights
            for i in range(num_neurons):
                for j in range(num_neurons):
                    neuron_pos = np.array([i, j])
                    dist_to_bmu = euclidean_distance(neuron_pos, bmu_idx)
                    if dist_to_bmu <= raduis:
                        Neighborhood = np.exp(-dist_to_bmu ** 2 / (2 * raduis ** 2))
                        current_error = vector - weights[i, j]
                        weights[i, j] += Neighborhood * learning_rate * current_error
        
        # Decay raduis and learning rate
        raduis = decay_function(raduis, t, num_iterations)
        learning_rate = decay_function(learning_rate, t, num_iterations)

train_som(data, weights, num_iterations, raduis, learning_rate)

def plot_som_with_labels(weights, data, labels):
    plt.figure(figsize=(10, 10))
    label_map = np.zeros((num_neurons, num_neurons, 10))  # To count the labels in each neuron
    for i, x in enumerate(data):
        bmu_idx = np.array([0, 0])
        min_dist = np.inf
        for i in range(num_neurons):
            for j in range(num_neurons):
                dist = euclidean_distance(x, weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = np.array([i, j])
        label_map[bmu_idx[0], bmu_idx[1], int(labels[i])] += 1
    
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

# Plot the result
plot_som_with_labels(weights, data, digits_keys)
