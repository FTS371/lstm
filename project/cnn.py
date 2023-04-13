import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读入数据
data = pd.read_csv('Data/Shanghai(20230316).csv')
# 提取收盘价
close = data['close'].values.reshape(-1,1)

# 数据预处理
scaler = MinMaxScaler()
close_scaled = scaler.fit_transform(close)

# 创建训练集和测试集
train_size = int(len(close_scaled) * 0.8)
test_size = len(close_scaled) - train_size
train_data = close_scaled[0:train_size,:]
test_data = close_scaled[train_size:len(close_scaled),:]

# 创建数据集
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:i+time_steps])
        y.append(dataset[i+time_steps])
    return np.array(X), np.array(y)

# 设置时间步长
time_steps = 30

# 创建训练集
X_train, y_train = create_dataset(train_data, time_steps)

# 创建测试集
X_test, y_test = create_dataset(test_data, time_steps)

# 调整数据形状
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_steps,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# 预测测试集
y_pred = model.predict(X_test)

# 反归一化处理
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)

# 评估预测结果
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
corr_coef = np.corrcoef(y_test.T, y_pred.T)[0][1]
print('RMSE:', rmse)
print('MAE:', mae)
print('R2:', r2)
print('Correlation coefficient:', corr_coef)

# 绘制预测图表
plt.figure(figsize=(16,8))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
