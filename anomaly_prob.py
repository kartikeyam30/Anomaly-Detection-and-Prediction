import keras.models
import numpy as np
import matplotlib.pyplot as plt
import signal
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import keras.backend as K
import tensorflow as tf

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
tf.compat.v1.enable_eager_execution()

dbscan_labels = np.load('final_models/anomaly/dbscan_5.npy')
dbscan_labels2 = np.load('final_models/anomaly/dbscan_all.npy')
forest_labels = np.load('final_models/anomaly/forest_5.npy')
forest_labels2 = np.load('final_models/anomaly/forest_all.npy')
encoder_labels = np.load('final_models/anomaly/autoencoder.npy')

forest_labels = np.where(forest_labels != -1, 0, 1)
dbscan_labels = np.where(dbscan_labels != -1, 0, 1)
forest_labels2 = np.where(forest_labels2 != -1, 0, 1)
dbscan_labels2 = np.where(dbscan_labels2 != -1, 0, 1)
encoder_labels = np.where(encoder_labels == False, 0, 1)

lab1 = dbscan_labels + forest_labels
lab2 = dbscan_labels2 + forest_labels2

# Calculating summed array
final_scores = lab1*1.5 + lab2*1 + encoder_labels*2
final_scores = final_scores/final_scores.max()
# %% Smoothing and probability calculation


def apply_smoothing_kernel(arr, kernel):
    # Apply the kernel to each row separately
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=arr)


def gaussian_kernel(size):
    sigma = 1
    # Create a vector from -size/2 to size/2
    range_vec = np.linspace(-1, 1, size)
    # Apply the Gaussian function
    kernel = np.exp(-0.5 * (range_vec / sigma) ** 2)
    kernel = kernel / np.max(kernel)
    return kernel


def smooth_and_aggregate_scores(final_scores, window_size=1):
    # Get unique non-zero values to process
    unique_values = np.unique(final_scores[final_scores != 0])

    # Initialize an array to hold the summed probabilities
    combined_probabilities = np.zeros_like(final_scores, dtype=float)

    # Smooth the array for each unique value and accumulate the probabilities
    for value in unique_values:
        # Create a mask for the current value
        value_mask = (final_scores == value).astype(float)

        # Smooth the mask with the kernel
        smoothed_mask = apply_smoothing_kernel(value_mask, gaussian_kernel(window_size))

        # Multiply by the value to restore the original scale
        combined_probabilities += smoothed_mask * value

    # Normalize the combined probabilities to [0, 1]
    max_prob = np.max(combined_probabilities)
    if max_prob > 0:
        final_probabilities = combined_probabilities / max_prob
    else:
        final_probabilities = combined_probabilities

    return final_probabilities


# Use the function on your final_scores
final_probabilities = smooth_and_aggregate_scores(final_scores, window_size=10)
np.save('anomalies/final_scores', final_probabilities)
