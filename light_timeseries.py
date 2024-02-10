import numpy as np
import matplotlib.pyplot as plt
import signal
import os
import datetime
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, \
    Input, BatchNormalization, Bidirectional, Permute, Reshape, Lambda, Multiply, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, Adamax, RMSprop, Adagrad, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# all = np.load('X_all.npy', allow_pickle=True)[:][:, -1]
X_or = np.load('X_data_500.npy')
Y_or = np.load('Y_data_500.npy')
probs = np.load('anomalies/final_scores.npy')

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
# %% LSTM Model training

train = int(len(X) * 0.7)
val = int(len(X) * 0.1)

X_train, X_test = np.array(X[:train + val, :, :-1], dtype=np.float16), np.array(X[train + val:, :, :-1], dtype=np.float16)
y_train, y_test = np.array(Y[:train + val, :48, -3], dtype=np.float16), \
                  np.array(Y[train + val:, :48, -3], dtype=np.float16)


def handle_interrupt(signum, frame):
    path = os.path.join('forecasting', datetime.date.strftime(datetime.date.today(), '%d_%m_%y'))
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_weights(os.path.join(path, 'lstm_weights.h5'))


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

reg = tf.keras.regularizers.L1L2(l1=1e-7, l2=1e-5)


def improved_lstm_model(array_in):
    input = Input(shape=(array_in.shape[1:]), name='input')
    i = 0
    lstm = Bidirectional(LSTM(512, activation='tanh', recurrent_activation='sigmoid',
                               return_sequences=True, name=f'lstm_1',
                               kernel_regularizer=reg,
                               recurrent_regularizer=reg, bias_regularizer=reg, dtype=np.float16))(input)
    norm = BatchNormalization()(lstm)
    drop = Dropout(0.4)(norm)

    lstm2 = Bidirectional(LSTM(256, activation='tanh', recurrent_activation='sigmoid',
                               return_sequences=True, name=f'lstm_3',
                               kernel_regularizer=reg,
                               recurrent_regularizer=reg, bias_regularizer=reg, dtype=np.float16))(drop)
    norm = BatchNormalization()(lstm2)
    drop2 = Dropout(0.4)(norm)
    lstm3 = Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                               return_sequences=True, name=f'lstm_4',
                               kernel_regularizer=reg,
                               recurrent_regularizer=reg, bias_regularizer=reg, dtype=np.float16))(drop2)
    norm = BatchNormalization()(lstm3)
    drop3 = Dropout(0.4)(norm)

    lstm3 = Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                               name=f'lstm_5', kernel_regularizer=reg,
                               recurrent_regularizer=reg, bias_regularizer=reg, dtype=np.float16))(drop3)
    norm = BatchNormalization()(lstm3)
    drop2 = Dropout(0.4)(norm)
    # drop2 = Dropout(0.4)(lstm2)

    x1 = Dense(48, name=f'output_var_{i}', dtype=np.float32)(drop2)

    model = Model(inputs=input, outputs=x1)
    return model


def train_and_evaluate(model, X_train, y_train, batch, epochs, callbacks):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        verbose=1,
        batch_size=batch,
        callbacks=callbacks,
        validation_split=0.3
    )
    return history


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9, staircase=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1,
                              min_delta=1e-4, cooldown=10)
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           restore_best_weights=True, min_delta=1e-4)
checkpoint = ModelCheckpoint('best_model_48.h5', monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min', )
callbacks = [checkpoint, early_stop, reduce_lr]

model = improved_lstm_model(X_train)
model.compile(optimizer=Adam(learning_rate=lr_schedule, clipnorm=1e-4),
              loss='mean_absolute_error',
              metrics=['mean_squared_error', tf.keras.metrics.R2Score()])

model.summary()
i = 0

steps = 10
epochs = 100

history = train_and_evaluate(model, X_train[:], y_train[:].astype(np.float32), steps, epochs, callbacks)

# %%
# input = np.concatenate((input[:, 1:, :], y_train[:, step, :].
# reshape((y_train.shape[0], 1, y_train.shape[2]))), axis=1)
# val_in = np.concatenate((val_in[:, 1:, :], y_val[:, step, :].reshape((y_val.shape[0], 1, y_val.shape[2]))), axis=1)

lr = history.history['lr']
loss += history.history['loss']
val_loss += history.history['val_loss']
mse += history.history['mean_squared_error']
val_mse += history.history['val_mean_squared_error']
acc += history.history['r2_score']
val_acc += history.history['val_r2_score']
# msle += history.history['huber_loss']
# val_msle += history.history['val_huber_loss']
# hinge_loss += history.history['hinge']
# val_hinge_loss += history.history['val_hinge']

K.clear_session()

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(36, 15))

metrics = [loss, mse, acc, lr]
metrics_val = [val_loss, val_mse, val_acc, lr]

for i in range(len(metrics)):
    if i < len(metrics)-1:
        axs.flatten()[i].plot(metrics[i], label=f'Training Loss {list(history.history.keys())[i]}', marker='o')
        axs.flatten()[i].plot(metrics_val[i],
                              label=f'Validation Loss {list(history.history.keys())[i + len(metrics)]}', marker='o')
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
tf.keras.models.save_model(model, 'final_models/anomaly/lstm.keras')
# %%
test_in = X_test  # [:, :, :12]
# for step in range(25):
y_pred = model.predict(test_in)

print(f'Test MAE : {np.mean(tf.keras.metrics.mean_absolute_error(y_test, y_pred))}')
print(f'Test MSE : {np.mean(tf.keras.metrics.mean_squared_error(y_test, y_pred))}')
print(f'Test R2 Score : {r2_score(y_test, y_pred)}')

'''if step == 0:
    pred = np.array(y_pred).reshape(X_test.shape[0], 1, 12)
else:
    pred = np.concatenate((pred, np.array(y_pred).reshape(X_test.shape[0], 1, 12)), axis=1)
test_in = np.concatenate((test_in[:, 1:, :12], np.array(y_pred).reshape(X_test.shape[0], 1, 12)), axis=1)'''

'''col = np.random.randint(0, 12)
df = np.random.randint(0, len(X_test))
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))
plt.plot(y_test[df, :])
plt.plot(y_pred[df, :])
axs[1].plot(y_test[df, :50, 11])
axs[1].plot(pred[df, :, 11])
plt.tight_layout()
plt.show()

old_pred = pred'''

# First, we'll create an index for the X values
index_X = range(300)

# The Y values continue where X left off, so its index starts after the last X index
index_Y = range(300, 348)

# The pred index will start where Y starts because it's a continuation of the series
index_pred = range(300, 348)

df = np.random.randint(0, len(X_test))
# Now let's plot X and Y as a continuous line
plt.figure(figsize=(12, 5))
plt.plot(index_X, X_test[df, -300:, -1], label='X - Historical W')
plt.plot(index_Y, y_test[df, :, 0], label='Y - True Future W')
'''
plt.plot(index_X, X_test[df, -300:, -1], label='X - Historical VA')
plt.plot(index_Y, y_test[df, :, 1], label='Y - True Future VA')'''
# And plot pred starting from the end of X
plt.plot(index_pred, y_pred[df], label='pred - Predicted W', linestyle='--', color='purple')
#plt.plot(index_pred, y_pred[1][df, :], label='pred - Predicted VA', linestyle='--', color='orange')

# Add some labels and a legend
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Time Series Forecast')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

# %%
import h5py
from io import BytesIO


def deepcopy_model_in_memory(model):
    # Save the model to a memory buffer
    model_copy = tf.keras.models.clone_model(model)

    # Set the weights to the clone
    model_copy.set_weights(model.get_weights())

    return model_copy


model_save = deepcopy_model_in_memory(model)
model_save.compile(optimizer=Adam(5e-4), loss='mse')


# %%
plt.figure(figsize=(100, 20))
plt.plot(np.array([X[a, :, -1] for a in range(len(X)) if X[a, :, -1].std() > 0.08]).T)
plt.xticks(range(695))
# plt.plot([X[a, :, -1] for a in range(len(X_test))])
# plt.plot(X_test[118, :, -1])
plt.show()