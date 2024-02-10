import numpy as np
import matplotlib.pyplot as plt
import signal
import copy
import os
import datetime
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import keras.backend as K
import tensorflow as tf

import gc

policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)
tf.compat.v1.enable_eager_execution()

# all = np.load('X_all.npy', allow_pickle=True)[:][:, -1]
X_or = np.load('X_data_500.npy')
Y_or = np.load('Y_data_500.npy')
probs = np.load('anomalies/final_scores.npy')

column_index = 1
threshold = 3

X2 = copy.deepcopy(X_or)

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

X = np.concatenate((X, probs[:, :-100].reshape(X.shape[0], X.shape[1], 1)), axis=2)
Y = np.concatenate((Y, probs[:, :100].reshape(Y.shape[0], Y.shape[1], 1)), axis=2)
# %% LSTM Based Transformer

train = int(len(X) * 0.7)
val = int(len(X) * 0.1)

X_train, X_test = np.array(X[:train + val, :, :-1], dtype=np.float16), np.array(X[train + val:, :, :-1], dtype=np.float16)
y_train, y_test = np.array(Y[:train + val, :48, -2], dtype=np.float16), \
                  np.array(Y[train + val:, :48, -2], dtype=np.float16)

kfold = KFold(n_splits=3, shuffle=True)

reg = tf.keras.regularizers.L1L2(l1=1e-7, l2=1e-5)


def handle_interrupt(signum, frame):
    path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'))
    if not os.path.exists(path):
        os.makedirs(path)
    transformer_model.save_weights(os.path.join(path, 'model_weights.h5'))


signal.signal(signal.SIGINT, handle_interrupt)
# %%
lr = []
loss = []
val_loss = []
hinge_loss = []
val_hinge_loss = []
mse = []
val_mse = []
acc = []
val_acc = []

time_steps = X_train.shape[1]
features = X_train.shape[2]

def create_look_back_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = inputs
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
    return x


def build_transformer_model(time_steps, features, head_size, num_heads, ff_dim, dropout=0.1):
    inputs = tf.keras.Input(shape=(time_steps, features), dtype=tf.float32)

    x = PositionalEncoding(features=features)(inputs)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True))(x)

    # Encoder stack
    for _ in range(2):  # Adjust the number of layers as needed
        x = transformer_encoder(lstm, head_size, num_heads, ff_dim, dropout)

    # Final processing and output
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(ff_dim/2), activation="tanh"))(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = tf.keras.layers.Dense(48, activation='linear')(x)  # Predicting the next 48 time points

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Custom class for positional encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.features = features

    def get_angles(self, pos, i):
        angles = 1 / tf.pow(10000.00, (2 * (i // 2)) / tf.cast(self.features, tf.float32))
        return pos * angles

    def positional_encoding(self, time_steps):
        angle_rads = self.get_angles(
            pos=tf.range(time_steps, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(self.features, dtype=tf.float32)[tf.newaxis, :])
        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        time_steps = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(time_steps)
        return inputs + pos_encoding[:, :time_steps, :]
# Compile the model
# transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')

# Print the model summary
# transformer_model.summary()

def gaussian_likelihood_loss(y_true, y_pred):
    # y_pred - [batch_size, time_steps, 2] - last dimension has mean and log variance
    assert y_pred.shape[1] == 2 * y_true.shape[1]
    mu, log_sigma = tf.split(y_pred, 2, axis=-1)

    sigma = tf.exp(log_sigma)

    y_true = tf.reshape(y_true, tf.shape(mu))

    loss = 0.5 * tf.reduce_mean(tf.math.log(sigma) + tf.square(y_true - mu) / sigma)
    return loss


transformer_model = build_transformer_model(time_steps=X_train.shape[1], features=X_train.shape[2],
                                            head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

# %% VA first run
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                             decay_steps=1000, decay_rate=0.96,
                                                             staircase=True)
transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                          loss='mean_absolute_error',
                          metrics=['mean_squared_error', tf.keras.metrics.R2Score()])
transformer_model.summary()

# for train, val in kfold.split(X_train, y_train):

logs = "logs/app_power/"

# specify the log directory
# board_callback = tf.keras.callbacks.TensorBoard(log_dir=logs + '1/',
#                                                histogram_freq=1,
#                                                profile_batch='10, 19')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
                                                 min_delta=1e-4, cooldown=5)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                                              restore_best_weights=True, min_delta=1e-4)
path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'))
checkpoint = tf.keras.callbacks.ModelCheckpoint(path+'/best_model_transformer_VA1.h5', monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='min')
callbacks = [reduce_lr, early_stop, checkpoint]

history_t1 = transformer_model.fit(X_train, y_train, batch_size=10, epochs=100,
                                  validation_split=0.3, callbacks=callbacks)
lr = history_t1.history['lr']
loss += history_t1.history['loss']
val_loss += history_t1.history['val_loss']
mse += history_t1.history['mean_squared_error']
val_mse += history_t1.history['val_mean_squared_error']
acc += history_t1.history['r2_score']
val_acc += history_t1.history['val_r2_score']
# msle += history_t.history['huber_loss']
# val_msle += history_t.history['val_huber_loss']
# hinge_loss += history_t.history['hinge']
# val_hinge_loss += history_t.history['val_hinge']

K.clear_session()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(36, 15))

metrics = [loss, mse, acc, lr]
metrics_val = [val_loss, val_mse, val_acc, lr]

for i in range(len(metrics)):
    if i < len(metrics)-1:
        axs.flatten()[i].plot(metrics[i], label=f'Training Loss {list(history_t1.history.keys())[i]}', marker='o')
        axs.flatten()[i].plot(metrics_val[i],
                              label=f'Validation Loss {list(history_t1.history.keys())[i + len(metrics)]}', marker='o')
        axs.flatten()[i].set_title('Training and Validation Loss')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
    else:
        axs.flatten()[i].plot(metrics_val[i], label=f'Learning Rate', marker='o')
        axs.flatten()[i].set_title('Learning Rate')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
plt.tight_layout()
plt.show()

# %% VA 2nd run
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.96,
                                                             staircase=True)
transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                          loss='mean_absolute_error',
                          metrics=['mean_squared_error', tf.keras.metrics.R2Score()])
transformer_model.summary()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                                 min_delta=1e-4, cooldown=5)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35, verbose=1,
                                              restore_best_weights=True, min_delta=1e-4)
path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'))
checkpoint = tf.keras.callbacks.ModelCheckpoint(path+'/best_model_transformer_VA2.h5', monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='min')
callbacks = [reduce_lr, early_stop, checkpoint]

history_t1 = transformer_model.fit(X_train, y_train, batch_size=3, epochs=20,
                                  validation_split=0.3, callbacks=callbacks)
# input = np.concatenate((input[:, 1:, :], y_train[:, step, :].
# reshape((y_train.shape[0], 1, y_train.shape[2]))), axis=1)
# val_in = np.concatenate((val_in[:, 1:, :], y_val[:, step, :].reshape((y_val.shape[0], 1, y_val.shape[2]))), axis=1)
lr = history_t1.history['lr']
loss += history_t1.history['loss']
val_loss += history_t1.history['val_loss']
mse += history_t1.history['mean_squared_error']
val_mse += history_t1.history['val_mean_squared_error']
acc += history_t1.history['r2_score']
val_acc += history_t1.history['val_r2_score']
# msle += history_t.history['huber_loss']
# val_msle += history_t.history['val_huber_loss']
# hinge_loss += history_t.history['hinge']
# val_hinge_loss += history_t.history['val_hinge']

K.clear_session()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(36, 15))

metrics = [loss, mse, acc, lr]
metrics_val = [val_loss, val_mse, val_acc, lr]

for i in range(len(metrics)):
    if i < len(metrics)-1:
        axs.flatten()[i].plot(metrics[i], label=f'Training Loss {list(history_t1.history.keys())[i]}', marker='o')
        axs.flatten()[i].plot(metrics_val[i],
                              label=f'Validation Loss {list(history_t1.history.keys())[i + len(metrics)]}', marker='o')
        axs.flatten()[i].set_title('Training and Validation Loss')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
    else:
        axs.flatten()[i].plot(metrics_val[i], label=f'Learning Rate', marker='o')
        axs.flatten()[i].set_title('Learning Rate')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
plt.tight_layout()
plt.show()

# %%
transformer_model = tf.keras.models.load_model('forecasting/21_11_23/best_model_transformer_VA1.h5',
                                   custom_objects={'PositionalEncoding': PositionalEncoding})
#%%
test_in = X_test  # [:, :, :12]
# for step in range(25):
y_pred = transformer_model.predict(test_in)

print(np.mean(tf.keras.metrics.mean_absolute_error(y_test, y_pred)))
print(np.mean(tf.keras.metrics.mean_squared_error(y_test, y_pred)))
print(np.mean(r2_score(y_test, y_pred)))

index_X = range(300)

index_Y = range(300, 348)

index_pred = range(300, 348)

path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'), 'graphs_VA')
os.makedirs(path)

for _ in range(25):
    df = np.random.randint(0, len(X_test))
    plt.figure(figsize=(20, 6))
    plt.plot(index_X, X_test[df, -300:, -2], label='X - Historical VA')
    plt.plot(index_Y, y_test[df, :], label='Y - True Future VA')
    '''
    plt.plot(index_X, X_test[df, -300:, -1], label='X - Historical VA')
    plt.plot(index_Y, y_test[df, :, 1], label='Y - True Future VA')'''
    plt.plot(index_pred, y_pred[df], label='pred - Predicted VA', linestyle='--', color='purple')
    # plt.plot(index_pred, y_pred[1][df, :], label='pred - Predicted VA', linestyle='--', color='orange')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Time Series Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'df_'+str(df)))
    plt.close()

# %%

y_pred1 = transformer_model.predict(X_train)
y_pred2 = transformer_model.predict(X_test)

# %%

lr2 = []
loss2 = []
val_loss2 = []
hinge_loss2 = []
val_hinge_loss2 = []
mse2 = []
val_mse2 = []
acc2 = []
val_acc2 = []

X_train, X_test = np.array(X[:train + val, :, :-1], dtype=np.float16), \
                  np.array(X[train + val:, :, :-1],dtype=np.float16)
y_train, y_test = np.array(Y[:train + val, :48, -3], dtype=np.float16), \
                  np.array(Y[train + val:, :48, -3], dtype=np.float16)

transformer_model2 = build_transformer_model(time_steps=X_train.shape[1], features=X_train.shape[2],
                                             head_size=128, num_heads=5, ff_dim=256, dropout=0.1)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                             decay_steps=1000, decay_rate=0.96,
                                                             staircase=True)
transformer_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.01),
                           loss='mean_absolute_error',
                           metrics=['mean_squared_error', tf.keras.metrics.R2Score()])
transformer_model2.summary()

# %%
logs = "logs/power/"

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                                 min_delta=1e-4, cooldown=5)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35, verbose=1,
                                              restore_best_weights=True, min_delta=1e-4)
path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'))
checkpoint = tf.keras.callbacks.ModelCheckpoint(path+'/best_model_transformer_W.h5', monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='min')
callbacks = [reduce_lr, early_stop, checkpoint]

history_t = transformer_model2.fit(X_train, y_train, batch_size=5, epochs=30,
                                   validation_split=0.3, callbacks=callbacks)
'''history_t = transformer_model.fit(X_train, y_train, batch_size=5, epochs=50,
                                  validation_split=0.3, callbacks=callbacks)'''
# input = np.concatenate((input[:, 1:, :], y_train[:, step, :].
# reshape((y_train.shape[0], 1, y_train.shape[2]))), axis=1)
# val_in = np.concatenate((val_in[:, 1:, :], y_val[:, step, :].reshape((y_val.shape[0], 1, y_val.shape[2]))), axis=1)
lr2 = history_t.history['lr']
loss2 += history_t.history['loss']
val_loss2 += history_t.history['val_loss']
mse2 += history_t.history['mean_squared_error']
val_mse2 += history_t.history['val_mean_squared_error']
acc2 += history_t.history['r2_score']
val_acc2 += history_t.history['val_r2_score']
# msle += history_t.history['huber_loss']
# val_msle += history_t.history['val_huber_loss']
# hinge_loss += history_t.history['hinge']
# val_hinge_loss += history_t.history['val_hinge']

K.clear_session()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(36, 15))

metrics = [loss2, mse2, acc2, lr2]
metrics_val = [val_loss2, val_mse2, val_acc2, lr2]

for i in range(len(metrics)):
    if i < len(metrics)-1:
        axs.flatten()[i].plot(metrics[i], label=f'Training Loss', marker='o')
        axs.flatten()[i].plot(metrics_val[i],
                              label=f'Validation Loss', marker='o')
        if list(history_t.history.keys())[i] == 'loss':
            axs.flatten()[i].set_title('mean_absolute_error')
        else:
            axs.flatten()[i].set_title(list(history_t.history.keys())[i])
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
    else:
        axs.flatten()[i].plot(metrics_val[i], label=f'Learning Rate', marker='o')
        axs.flatten()[i].set_title('Learning Rate')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
plt.tight_layout()
plt.show()

# %%
transformer_model2 = tf.keras.models.load_model('forecasting/21_11_23/best_model_transformer_W.h5',
                                                custom_objects={'PositionalEncoding': PositionalEncoding})

# %%
test_in = X_test  # [:, :, :12]
# for step in range(25):
y_pred = transformer_model2.predict(test_in)

print(np.mean(tf.keras.metrics.mean_absolute_error(y_test, y_pred)))
print(np.mean(tf.keras.metrics.mean_squared_error(y_test, y_pred)))
print(np.mean(r2_score(y_test, y_pred)))

df = np.random.randint(0, len(X_test))
plt.figure(figsize=(20, 6))
plt.plot(index_X, X_test[df, -300:, -3], label='X - Historical W')
plt.plot(index_Y, y_test[df, :], label='Y - True Future W')
plt.plot(index_pred, y_pred[df], label='pred - Predicted W', linestyle='--', color='purple')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Time Series Forecast')
plt.legend()
plt.tight_layout()
plt.show()


# %%
plt.boxplot([np.std(X_test[:, :, -2], axis=1), np.std(X_test[:, :, -3], axis=1)])
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(np.std(X_test[:, :, -2], axis=1)/np.std(X_test[:, :, -2], axis=1).max(), label='VA')
plt.plot(np.std(X_test[:, :, -3], axis=1)/np.std(X_test[:, :, -3], axis=1).max(), label='W')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 6))
plt.plot(X_test[576, :, -3], label='W')
plt.plot(X_test[576, :, -2], label='VA')
plt.plot(X_test[576, :, -1], label='Prob')
plt.legend()
plt.tight_layout()
plt.show()
# %%

y_pred1_W = transformer_model2.predict(X_train)
y_pred2_W = transformer_model2.predict(X_test)

# %%


lr3 = []
loss3 = []
val_loss3 = []
hinge_loss3 = []
val_hinge_loss3 = []
mse3 = []
val_mse3 = []
acc3 = []
val_acc3 = []

X_train, X_test = np.array(X[:train + val], dtype=np.float16), np.array(X[train + val:], dtype=np.float16)
y_train, y_test = np.array(Y[:train + val, :48, -1], dtype=np.float16), \
                  np.array(Y[train + val:, :48, -1], dtype=np.float16)

transformer_model3 = build_transformer_model(time_steps=X_train.shape[1], features=X_train.shape[2],
                                             head_size=256, num_heads=4, ff_dim=128, dropout=0.2)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=5000, decay_rate=0.96,
                                                             staircase=True)
transformer_model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                           loss='mean_absolute_error',
                           metrics=['mean_squared_error', tf.keras.metrics.R2Score()])
transformer_model3.summary()

# for train, val in kfold.split(X_train, y_train):

# %%
logs = "logs/power/"  # specify the log directory
# board_callback = tf.keras.callbacks.TensorBoard(log_dir=logs + '1/',
#                                                histogram_freq=1,
#                                                profile_batch='10, 19')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1,
                                                 min_delta=1e-4, cooldown=5)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True,
                                              min_delta=1e-4)
path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'))
checkpoint = tf.keras.callbacks.ModelCheckpoint(path+'/best_model_transformer_prob.h5', monitor='val_loss',
                                                verbose=1, save_best_only=True, mode='min')
callbacks = [reduce_lr, early_stop, checkpoint]

history_t = transformer_model3.fit(X_train, y_train, batch_size=10, epochs=100,
                                   validation_split=0.3, callbacks=callbacks)
# input = np.concatenate((input[:, 1:, :], y_train[:, step, :].
# reshape((y_train.shape[0], 1, y_train.shape[2]))), axis=1)
# val_in = np.concatenate((val_in[:, 1:, :], y_val[:, step, :].reshape((y_val.shape[0], 1, y_val.shape[2]))), axis=1)
lr3 = history_t.history['lr']
loss3 += history_t.history['loss']
val_loss3 += history_t.history['val_loss']
mse3 += history_t.history['mean_squared_error']
val_mse3 += history_t.history['val_mean_squared_error']
acc3 += history_t.history['r2_score']
val_acc3 += history_t.history['val_r2_score']
# msle += history_t.history['huber_loss']
# val_msle += history_t.history['val_huber_loss']
# hinge_loss += history_t.history['hinge']
# val_hinge_loss += history_t.history['val_hinge']

K.clear_session()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(36, 15))

metrics3 = [loss3, mse3, acc3, lr3]
metrics_val3 = [val_loss3, val_mse3, val_acc3, lr3]

for i in range(len(metrics)):
    if i < len(metrics)-1:
        axs.flatten()[i].plot(metrics3[i], label=f'Training Loss {list(history_t.history.keys())[i]}', marker='o')
        axs.flatten()[i].plot(metrics_val3[i],
                              label=f'Validation Loss {list(history_t.history.keys())[i + len(metrics3)]}', marker='o')
        axs.flatten()[i].set_title('Training and Validation Loss')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
    else:
        axs.flatten()[i].plot(metrics_val3[i], label=f'Learning Rate', marker='o')
        axs.flatten()[i].set_title('Learning Rate')
        axs.flatten()[i].set_xlabel('Epochs')
        axs.flatten()[i].legend()
        axs.flatten()[i].set_ylabel('Loss')
        axs.flatten()[i].grid(True)
plt.tight_layout()
plt.show()

# %%
test_in = X_test  # [:, :, :12]
# for step in range(25):
y_pred = transformer_model3.predict(test_in)

print(np.mean(tf.keras.metrics.mean_absolute_error(y_test, y_pred)))
print(np.mean(tf.keras.metrics.mean_squared_error(y_test, y_pred)))
print(np.mean(r2_score(y_test, y_pred)))

df = np.random.randint(0, len(X_test))
plt.figure(figsize=(20, 6))
plt.plot(range(495), X_test[df, :, -1], label='X - Historical Probability')
plt.plot(range(495, 495+48), y_test[df, :], label='Y - True Future Probability')
plt.plot(range(495, 495+48), y_pred[df], label='pred - Predicted Probability', linestyle='--', color='purple')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Time Series Forecast')
plt.legend()
plt.tight_layout()
plt.show()


# %%
y_pred1_prob1 = transformer_model3.predict(X_train)
y_pred1_prob2 = transformer_model3.predict(X_test)

# %%

pred_VA = np.concatenate((y_pred1, y_pred2), axis=0)
pred_W = np.concatenate((y_pred1_W, y_pred2_W), axis=0)
pred_prob = np.concatenate((y_pred1_prob1, y_pred1_prob2), axis=0)

#path = os.path.join('final', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'), 'graphs_overall')
#os.makedirs(path)

for i in range(50):
    df = np.random.randint(0, len(X))

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 18))
    #plt.cm.get_cmap('Paired')
    axs.flatten()[0].plot(index_X, X[df, -300:, -1], label='X - Historical Probability')
    axs.flatten()[0].plot(index_Y, Y[df, :48, -1], label='Y - True Future Probability')
    axs.flatten()[1].plot(index_X, X[df, -300:, -3], label='X - Historical Real Power')
    axs.flatten()[1].plot(index_Y, Y[df, :48, -3], label='Y - True Future Real Power')
    axs.flatten()[2].plot(index_X, X[df, -300:, -2], label='X - Historical Apparent Power')
    axs.flatten()[2].plot(index_Y, Y[df, :48, -2], label='Y - True Future Apparent Power')

    axs.flatten()[2].plot(index_pred, pred_VA[df], label='pred - Predicted Apparent Power', linestyle='--')
    axs.flatten()[1].plot(index_pred, pred_W[df], label='pred - Predicted Real Power', linestyle='--')
    axs.flatten()[0].plot(index_pred, pred_prob[df], label='pred - Predicted Probability', linestyle='--')
    for i in range(3):
        axs.flatten()[i].legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(path+'/df_'+str(df))
    plt.close()