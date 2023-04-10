import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据准备
df = pd.read_csv('stock_prices.csv')
train_size = int(len(df) * 0.7)
train_data = df[:train_size].copy()
test_data = df[train_size:].copy()
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# 2. CNN模型搭建
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 5)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

cnn_model.compile(optimizer='adam', loss='mse')

# 3. CNN模型训练
X_train, y_train = create_dataset(train_data, train_data[:, 1], 30)
cnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# 4. CNN模型预测
X_test, y_test = create_dataset(test_data, test_data[:, 1], 30)
y_pred_cnn = cnn_model.predict(X_test)
y_pred_cnn = scaler.inverse_transform(y_pred_cnn)
y_test = scaler.inverse_transform([y_test])

# 5. RNN模型搭建
rnn_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 5)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mse')

# 6. RNN模型训练
X_train, y_train = create_dataset(train_data, train_data[:, 1], 30)
rnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# 7. RNN模型预测
X_test, y_test = create_dataset(test_data, test_data[:, 1], 30)
y_pred_rnn = rnn_model.predict(X_test)
y_pred_rnn = scaler.inverse_transform(y_pred_rnn)
y_test = scaler.inverse_transform([y_test])

# 8. 图表输出
plt.figure(figsize=(16, 8))
plt.plot(y_test[0], label='True')
plt.plot(y_pred_cnn[:, 0], label='CNN Predicted')
plt.legend()
plt.title('CNN Stock Price Prediction')
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(y_test[0], label='True')
plt.plot(y_pred_rnn[:, 0], label='RNN Predict')
plt.legend()
plt.title('RNN Stock Price Prediction')
plt.show()
