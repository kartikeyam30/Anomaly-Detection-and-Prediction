import os
import random

import keras.models
import numpy as np
import matplotlib.pyplot as plt
import signal
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import keras.backend as K
import tensorflow as tf
import gc
import hdbscan
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
tf.compat.v1.enable_eager_execution()

# all = np.load('X_all.npy', allow_pickle=True)[:][:, -1]
X_or = np.load('X_data_500.npy')
Y_or = np.load('Y_data_500.npy')

scalers = {}
X = np.zeros_like(X_or)
Y = np.zeros_like(Y_or)
for i in range(X_or.shape[0]):
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(np.concatenate((X_or[i], Y_or[i]), axis=0))
    X[i] = transformed[:len(X_or[i])]
    Y[i] = transformed[len(X_or[i]):]
    scalers[i] = scaler

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

X = np.concatenate((X, Y), axis=1)
# %% HDBSCAN on 5 variables
def run_dbscan(data):
    dbscan = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=20, cluster_selection_epsilon=10,
                             allow_single_cluster=False, alpha=0.7, cluster_selection_method='leaf')
    clusters = dbscan.fit_predict(data)
    return dbscan, clusters, clusters == 1  # Return True for anomalies


dbscan_labels = []
dbscan_scores = []
dbscan_number = []
for sample in range(0, int(len(X)), 1):
    model_dbscan, cluster_dbscan, anomalies_dbscan = run_dbscan(X[sample, :, -5:])
    if min(anomalies_dbscan) == max(anomalies_dbscan):
        dbscan_labels.append(cluster_dbscan)
        continue
    else:
        num = len(cluster_dbscan[cluster_dbscan == -1])
        silhouette = silhouette_score(X[sample, :, -5:], anomalies_dbscan)
        if num > 50:
            cluster_dbscan = [1 for _ in range(595)]
            silhouette = np.mean(dbscan_scores)
        dbscan_number.append(num)
        dbscan_scores.append(silhouette)
        dbscan_labels.append(cluster_dbscan)

print(np.mean(dbscan_scores), np.median(dbscan_scores), np.min(dbscan_scores), np.max(dbscan_scores))
print(np.mean(dbscan_number), max(dbscan_number), min(dbscan_number))
print(np.count_nonzero(dbscan_number)/len(dbscan_number))


# %% HDBSCAN on all variables
def run_dbscan2(data):
    dbscan = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=4, cluster_selection_epsilon=1.05,
                             allow_single_cluster=False)
    clusters = dbscan.fit_predict(data)
    return dbscan, clusters, clusters == 1  # Return True for anomalies


dbscan_labels2 = []
dbscan_scores2 = []
dbscan_number2 = []
for sample in range(0, int(len(X)), 1):
    model_dbscan2, cluster_dbscan2, anomalies_dbscan2 = run_dbscan2(X[sample])
    if min(anomalies_dbscan2) == max(anomalies_dbscan2):
        dbscan_labels.append(cluster_dbscan2)
        continue
    else:
        num = len(cluster_dbscan2[cluster_dbscan2 == -1])
        silhouette = silhouette_score(X[sample], anomalies_dbscan2)
        if num > 50:
            cluster_dbscan = [1 for _ in range(595)]
            silhouette = np.mean(dbscan_scores2)
        dbscan_number2.append(num)
        dbscan_scores2.append(silhouette)
        dbscan_labels2.append(cluster_dbscan2)

print(np.mean(dbscan_scores2), np.median(dbscan_scores2), np.min(dbscan_scores2), np.max(dbscan_scores2))
print(np.mean(dbscan_number2), max(dbscan_number2), min(dbscan_number2))
print(np.count_nonzero(dbscan_number2) / len(dbscan_number2))


# %% iForest on 5 variables
def calculate_dynamic_contamination(data_segment, X, cols):
    segment_std = np.std(data_segment, axis=0)
    highest_std_observed = np.max(np.std(X[:, :, -cols:], axis=0), axis=0)  # Calculate this once and store it
    contamination_ratio = np.clip(segment_std / highest_std_observed, a_min=0, a_max=1)
    return np.percentile(contamination_ratio, 1)/15


def run_isolation_forest(data, X, cols, n_estimators=1500, features=5):
    contamination = calculate_dynamic_contamination(data, X, cols)
    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, n_jobs=-1,
                                 max_features=features, bootstrap=True)
    forest_clusters = iso_forest.fit_predict(data)
    decision_scores = iso_forest.decision_function(data)
    score_samples = iso_forest.score_samples(data)
    if len(forest_clusters[forest_clusters == -1]) > 20:
        print(contamination)
    return forest_clusters, forest_clusters == -1, decision_scores, score_samples


# Apply Isolation Forest to segments of the dataset
forest_labels = []
forest_scores_raw = []
forest_scores_raw_avg = []
forest_scores_anomaly = []
forest_scores_normal = []
forest_number = []

for sample in range(0, len(X)-1, 20):  # Loop through data in chunks
    for d in range(20):
        data_segment = X[d + sample, :, -5:]
        clusters, anomalies, dec_scores, score_samples = run_isolation_forest(data_segment, X, 5)

        forest_scores_normal.extend(dec_scores[~anomalies])
        forest_scores_anomaly.extend(dec_scores[anomalies])
        forest_scores_raw.extend(score_samples)
        forest_scores_raw_avg.extend(score_samples[anomalies])
        forest_labels.append(clusters)
        forest_number.append(np.sum(anomalies))

        if (sample == len(X)-1) and (d==19):
            break

# Print statistics
print(
    f"Anomaly Scores: Mean={np.mean(forest_scores_anomaly)}, Median={np.median(forest_scores_anomaly)}, "
    f"Max={np.max(forest_scores_anomaly)}, Min={np.min(forest_scores_anomaly)}")
print(
    f"Normal Scores: Mean={np.mean(forest_scores_normal)}, Median={np.median(forest_scores_normal)}, "
    f"Min={np.min(forest_scores_normal)}, Max={np.max(forest_scores_normal)}")
print(
    f"Raw Scores: Mean={np.mean(forest_scores_raw)}, Median={np.median(forest_scores_raw)}, "
    f"Min={np.min(forest_scores_raw)}, Max={np.max(forest_scores_raw)}")
print(
    f"Raw Scores Anomalies: Mean={np.mean(forest_scores_raw_avg)}, Median={np.median(forest_scores_raw_avg)}, "
    f"Min={np.min(forest_scores_raw_avg)}, Max={np.max(forest_scores_raw_avg)}")
print(f"Anomaly Count: Mean={np.mean(forest_number)}, Max={np.max(forest_number)}, Min={np.min(forest_number)}")


# %% iForest on all variables

forest_labels2 = []
forest_scores_raw2 = []
forest_scores_raw_avg2 = []
forest_scores_anomaly2 = []
forest_scores_normal2 = []
forest_number2 = []

for sample in range(0, len(X)-1, 20):  # Loop through data in chunks
    for d in range(20):
        data_segment = X[d + sample, :, :]
        clusters2, anomalies2, dec_scores2, score_samples2 = run_isolation_forest(data_segment, X, 35)

        forest_scores_normal2.extend(dec_scores2[~anomalies2])
        forest_scores_anomaly2.extend(dec_scores2[anomalies2])
        forest_scores_raw2.extend(score_samples2)
        forest_scores_raw_avg2.extend(score_samples2[anomalies2])
        forest_labels2.append(clusters2)
        forest_number2.append(np.sum(anomalies2))
        if (sample == len(X)-1) and (d==19):
            break

# Print statistics
print(
    f"Anomaly Scores: Mean={np.mean(forest_scores_anomaly2)}, Median={np.median(forest_scores_anomaly2)}, "
    f"Max={np.max(forest_scores_anomaly2)}, Min={np.min(forest_scores_anomaly2)}")
print(
    f"Normal Scores: Mean={np.mean(forest_scores_normal2)}, Median={np.median(forest_scores_normal2)}, "
    f"Min={np.min(forest_scores_normal2)}, Max={np.max(forest_scores_normal2)}")
print(
    f"Raw Scores: Mean={np.mean(forest_scores_raw2)}, Median={np.median(forest_scores_raw2)}, "
    f"Min={np.min(forest_scores_raw2)}, Max={np.max(forest_scores_raw2)}")
print(
    f"Raw Scores Anomalies: Mean={np.mean(forest_scores_raw_avg2)}, Median={np.median(forest_scores_raw_avg2)}, "
    f"Min={np.min(forest_scores_raw_avg2)}, Max={np.max(forest_scores_raw_avg2)}")
print(f"Anomaly Count: Mean={np.mean(forest_number2)}, Max={np.max(forest_number2)}, Min={np.min(forest_number2)}")

# %%
np.save('anomalies/forest_labels_new', forest_labels)
np.save('anomalies/dbscan_labels_new', dbscan_labels)
# %% LSTM Based Auto-Encoder


def build_lstm_autoencoder(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    encoded = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, activation='tanh',
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))(input_layer)
    x = tf.keras.layers.BatchNormalization()(encoded)
    x = tf.keras.layers.Dropout(0.5)(x)
    encoded = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh',
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))(x)
    x = tf.keras.layers.BatchNormalization()(encoded)
    x = tf.keras.layers.Dropout(0.5)(x)
    encoded = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(16, return_sequences=True, activation='tanh',
                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)))(x)
    x = tf.keras.layers.BatchNormalization()(encoded)
    x = tf.keras.layers.Dropout(0.5)(x)
    encoded = tf.keras.layers.LSTM(4, return_sequences=True, activation='tanh',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(encoded)
    decoded = tf.keras.layers.LSTM(16, return_sequences=True, activation='tanh')(x)
    x = tf.keras.layers.BatchNormalization()(decoded)
    decoded = tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh')(x)
    x = tf.keras.layers.BatchNormalization()(decoded)
    decoded = tf.keras.layers.LSTM(128, return_sequences=True, activation='tanh')(x)
    x = tf.keras.layers.BatchNormalization()(decoded)
    # Decoder
    decoded = tf.keras.layers.Dense(35, activation='sigmoid')(x)

    # Build the autoencoder model
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                        loss='mean_absolute_error', metrics=['mse', 'accuracy'])
    return autoencoder


autoencoder = build_lstm_autoencoder(X.shape[1:])
autoencoder.summary()

# Train Autoencoder
model = autoencoder.fit(X[:int(len(X) * 0.8)], X[:int(len(X) * 0.8)],
                        epochs=10, batch_size=50, shuffle=True, validation_split=0.2)
model = autoencoder.fit(X[:int(len(X) * 0.8)], X[:int(len(X) * 0.8)],
                        epochs=10, batch_size=35, shuffle=True, validation_split=0.2)
model = autoencoder.fit(X[:int(len(X) * 0.8)], X[:int(len(X) * 0.8)],
                        epochs=10, batch_size=20, shuffle=True, validation_split=0.2)
model = autoencoder.fit(X[:int(len(X) * 0.8)], X[:int(len(X) * 0.8)],
                        epochs=30, batch_size=15, shuffle=True, validation_split=0.2)

plt.figure(figsize=(10, 6))
plt.plot(model.history['loss'], label='loss')
plt.plot(model.history['val_loss'], label='val_loss')
plt.plot(model.history['accuracy'], linestyle='-.', label='acc')
plt.plot(model.history['val_accuracy'], linestyle='-.', label='val_acc')
plt.legend()
plt.tight_layout()
plt.show()

# tf.keras.models.save_model(autoencoder, 'anomalies/lstm_autoencoder.h5')
# %% Predictions from Auto-Encoder

autoencoder = tf.keras.models.load_model('anomalies/lstm_autoencoder.h5')
# Detect anomalies using reconstruction error
reconstructions = autoencoder.predict(X)  # .reshape(X[int(len(X) * 0.8):].shape[0], X.shape[1]*X.shape[2],))
# reconstructions = reconstructions.reshape(X[int(len(X) * 0.8):].shape[0], X.shape[1], X.shape[2])
reconstruction_errors = np.mean(np.abs(X - reconstructions), axis=2)
threshold = np.percentile(reconstruction_errors, 97.5, axis=1)
anomalies_autoencoder = np.array([reconstruction_errors[i] > threshold[i] for i in range(len(X))])

sample = np.random.randint(0, len(X))
plt.figure(figsize=(12, 5))
plt.plot(X[sample, :, -2], label='original')
plt.plot(reconstructions[sample, :, -2], label='predicted')
plt.scatter(np.where(anomalies_autoencoder[sample]==True),
            reconstructions[sample, anomalies_autoencoder[sample], -2], color='red', label='anomalies')
plt.legend()
plt.tight_layout()
plt.show()

np.save('anomalies/encoder_labels', anomalies_autoencoder)
# %% Analysing Anomaly Labelling

for i in range(1):
    plt.figure(figsize=(25, 12))
    sample = np.random.randint(0, len(X))
    # Plotting last two features of the random sample
    plt.plot(X[sample, :, -2], label='Feature 1')
    plt.plot(X[sample, :, -1], label='Feature 2')

    # Plotting anomalies detected by each method
    plt.scatter(np.where(anomalies_autoencoder[sample]),
                X[sample, anomalies_autoencoder[sample], -2],
                c='blue', s=90, label='Autoencoder Anomalies')
    plt.scatter(np.where(dbscan_labels[sample]==-1), X[sample, dbscan_labels[sample]==-1, -2],
                c='red', s=80, label='DBSCAN Anomalies')
    plt.scatter(np.where(dbscan_labels2[sample]==-1), X[sample, dbscan_labels2[sample]==-1, -2],
                c='black', s=70, label='DBSCAN Anomalies Old')
    plt.scatter(np.where(forest_labels2[sample] == -1), X[sample, forest_labels2[sample] == -1, -2],
                c='green', s=60, label='Isolation Forest Anomalies Old')
    plt.scatter(np.where(forest_labels[sample] == -1), X[sample, forest_labels[sample] == -1, -2],
                c='orange', s=50, label='Isolation Forest Anomalies')

    plt.legend()
    plt.title('Anomaly Detection Comparison')
    plt.tight_layout()
    plt.show()
    print(len(np.where(dbscan_labels[sample] == -1)[0]))
    # plt.savefig(f'anomalies/images/anomalies-{sample}.jpg')

# %% Saving all anomaly arrays
import os
path = 'final_models/anomaly'
os.makedirs(path)

np.save(path+'//dbscan_5', dbscan_labels)
np.save(path+'//dbscan_all', dbscan_labels2)
np.save(path+'//forest_5', forest_labels)
np.save(path+'//forest_all', forest_labels2)
np.save(path+'//autoencoder', anomalies_autoencoder)
