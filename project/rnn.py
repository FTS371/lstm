import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
df = pd.read_csv('Data/Shanghai(20230316).csv', parse_dates=['date'])
df = df[['date', 'close']]
df = df.set_index('date')

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# 创建时间序列数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 创建训练集和测试集
train_size = int(len(df_scaled) * 0.8)
test_size = len(df_scaled) - train_size
train, test = df_scaled[0:train_size,:], df_scaled[train_size:len(df_scaled),:]

# 用时间步创建X和Y
look_back = 30
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# 调整输入数据的维度
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建RNN模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 预测测试集
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform([y_test])

# 计算误差指标
rmse = np.sqrt(mean_squared_error(y_test[0], y_pred[:,0]))
mae = mean_absolute_error(y_test[0], y_pred[:,0])
r2 = r2_score(y_test[0], y_pred[:,0])
corr = np.corrcoef(y_test[0], y_pred[:,0])[0,1]

# 绘制预测图表
train_predict = model.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

# 构造用于绘制预测图表的日期索引
train_dates = df.index[look_back:train_size].values
test_dates = df.index[train_size+look_back:].values

# 绘制训练集和测试集的真实值和预测值
plt.plot(df.index, df.values, label='True value')
plt.plot(train_dates, train_predict, label='Train predict')
plt.plot(test_dates, test_predict, label='Test predict')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close price')
plt.title('Stock price prediction using LSTM')
plt.show()

# 输出误差指标
print('RMSE:', rmse)
print('MAE:', mae)
print('R2:', r2)
print('Correlation coefficient:', corr)
