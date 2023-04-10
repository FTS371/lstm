import numpy as np
import pandas as pd
import tensorflow as tf
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

time_steps = 30
X_train, y_train = create_dataset(train_data, train_data[:, 1], time_steps)
X_test, y_test = create_dataset(test_data, test_data[:, 1], time_steps)

# 2. 模型搭建
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_steps, 5)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 3. 模型训练
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# 4. 模型预测
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform([y_test])

# 5. 计算相关系数和平均误差
r2 = r2_score(y_test[0], y_pred[:, 0])
mse = mean_squared_error(y_test[0], y_pred[:, 0])

print('R2 Score: %.2f' % r2)
print('Mean Squared Error: %.2f' % mse)
