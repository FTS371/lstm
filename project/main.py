# # 导入所需的库
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import LSTM, Dense
# # 读取股票数据，这里以上汽公司为例
# df = pd.read_csv('Data/Shanghai(20230316).csv')# 只保留收盘价一列
# df = df[['close']]# 查看数据的前五行
# df.head()# 绘制收盘价走势图
# # plt.backend('TkAgg')
# plt.figure(figsize=(12,6))
# plt.plot(df)
# plt.title('Stock Price')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.show()
# # 将数据分为训练集和测试集，这里以80%为训练集，20%为测试集
# train_size = int(len(df) * 0.8)
# test_size = len(df) - train_size
# train_data = df.iloc[:train_size]
# test_data = df.iloc[train_size:]
# # 对数据进行归一化处理，使其在0到1之间变化
# scaler = MinMaxScaler()
# train_data = scaler.fit_transform(train_data)
# test_data = scaler.transform(test_data)
# # 定义一个函数，将数据转换为时间序列格式，即每个样本包含n个时间步的数据和一个标签（下一个时间步的数据）
# def create_dataset(data, n):
#     x = []
#     y = []
#     for i in range(len(data)-n-1):
#         x.append(data[i:i+n])
#         y.append(data[i+n])
#     return np.array(x), np.array(y)
# # 设置时间步为10，即每个样本包含10天的收盘价和第11天的收盘价作为标签
# n = 10
# x_train, y_train = create_dataset(train_data, n)
# x_test, y_test = create_dataset(test_data, n)
# # 查看训练集和测试集的形状
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# # 创建LSTM模型，包含一个LSTM层和一个全连接层，输出维度为1（预测值)
# model = Sequential()
# model.add(LSTM(500, input_shape=(n,1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# # 训练模型，设置迭代次数为50，批大小为32
# model.fit(x_train, y_train, epochs=50, batch_size=32)
# model.save('lstm_model.h5')  # 将模型保存到名为lstm_model.h5的文件中
# # 预测测试集，并将结果反归一化还原为原始价格范围
# y_pred = model.predict(x_test)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# # 计算预测误差的均方根（RMSE）
# rmse = np.sqrt(np.mean((y_pred - y_test)**2))
# print('RMSE:', rmse)
# # 绘制真实值和预测值的对比图
# plt.figure(figsize=(12,6))
# plt.plot(y_test, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.title('shanghai Stock Price Prediction')
# plt.xlabel('Days')
# plt.ylabel('Close Price')
# plt.legend()
# plt.show()

# #import necessary libraries
# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
#
# #Read in the stock data
# data = pd.read_csv('stock_data.csv')
#
# #Separate out the input and output
# features X = data.iloc[:, :-1].values y = data.iloc[:, -1].values
#
# #Split the data into training and testing
# sets X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
# #Create the LSTM model
# model = Sequential() model.add(LSTM(units = 50, activation = 'relu', input_shape = (X_train.shape[1], 1))) model.add(Dense(units = 1)) model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#
# #Reshape the input data
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
# #Train the LSTM model
# model.fit(X_train, y_train, epochs = 100, batch_size = 32)
#
# #Make predictions
# y_pred = model.predict(X_test)
# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取股票数据，这里以上汽公司为例
df = pd.read_csv('Data/Shanghai(20230316).csv')

# 只保留收盘价一列
df = df[['close']]

# 查看数据的前五行
df.head()

# 绘制收盘价走势图
plt.figure(figsize=(12,6))
plt.plot(df)
plt.title('Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# 将数据分为训练集和测试集，这里以80%为训练集，20%为测试集
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# 对数据进行归一化处理，使其在0到1之间变化
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 定义一个函数，将数据转换为时间序列格式，即每个样本包含n个时间步的数据和一个标签（下一个时间步的数据）
def create_dataset(data, n):
    x = []
    y = []
    for i in range(len(data)-n-1):
        x.append(data[i:i+n])
        y.append(data[i+n])
    return np.array(x), np.array(y)

# 设置时间步为10，即每个样本包含10天的收盘价和第11天的收盘价作为标签
n = 10
x_train, y_train = create_dataset(train_data, n)
x_test, y_test = create_dataset(test_data, n)

# 查看训练集和测试集的形状
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 创建LSTM模型，包含一个LSTM层和一个全连接层，输出维度为1（预测值)
model = Sequential()
model.add(LSTM(500, input_shape=(n,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型，设置迭代次数为50，批大小为32
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('lstm_model.h5')  # 将模型保存到名为lstm_model.h5的文件中

# 预测测试集，并将结果反归一化还原为原始价格范围
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# 计算预测误差的均方根（RMSE）、平均绝对误差（MAE）、决定系数（R2）和相关系数
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
corr_coef = np.corrcoef(y_test.T, y_pred.T)[0, 1]

print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)
print('R2:', r2)
print('Correlation Coefficient:', corr_coef)

# 绘制真实值和预测值的对比图
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('shanghai Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()
