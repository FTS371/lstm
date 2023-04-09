import pandas as pd
df = pd.read_csv('Data/shanghai.csv')
df = df[['open', 'high', 'low', 'close', 'volume']]
df.head()
# 读取股票数据，并保留开盘价、最高价、最低价、收盘价和成交量这五列
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
# 对数据进行归一化处理，使其在0到1之间变化。你可以使用sklearn.preprocessing中的MinMaxScaler类来实现
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data = df[:train_size]
test_data = df[train_size:]
# 定义一个函数，将数据转换为时间序列格式，即每个样本包含n个时间步的数据和一个标签（下一个时间步的收盘价）
def create_dataset(data, n):
    x = []
    y = []
    for i in range(len(data)-n-1):
        x.append(data[i:i+n])
        y.append(data[i+n][3]) # 取第四列作为标签（收盘价）
    return np.array(x), np.array(y)
# 设置时间步为10，并用上面定义的函数创建训练集和测试集
import numpy as np
n = 10
x_train, y_train = create_dataset(train_data, n)
x_test, y_test = create_dataset(test_data, n)
# 创建LSTM模型，并添加一个全连接层作为输出层。注意输入层的形状应该与x_train相匹配。
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(500, input_shape=(n,5))) # 输入层有5个特征（开盘价、最高价、最低价、收盘价和成交量）
model.add(Dense(5)) # 输出层只有一个值（预测值）
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('lstm_model.h5')  # 将模型保存到名为lstm_model.h5的文件中
# 预测测试集，并将结果反归一化还原为原始价格范围。注意反归一化时要用到原始数据中所有特征的最大值和最小值
y_pred = model.predict(x_test)

# 创建一个空数组用于存放反归一化后的预测值和真实值
y_pred_inv = np.zeros((len(y_pred), 5))
y_test_inv = np.zeros((len(y_test), 5))

# 将预测值和真实值放在第四列（对应收盘价）
y_pred_inv[:,3] = y_pred[:,0]
y_test_inv[:,3] = y_test

# 使用scaler.inverse_transform方法进行反归一化，并取出第四列作为结果
y_pred_inv = scaler.inverse_transform(y_pred_inv)[:,3]
y_test_inv = scaler.inverse_transform(y_test_inv)[:,3]



# 使用上面训练好的LSTM模型，并用测试集中最后10个时间步作为输入，预测下一个时间步作为输出
x_input = x_test[-1]
y_output = model.predict(x_input.reshape(1,n,5))

# 将预测值添加到测试集中，并将其反归一化还原为原始价格范围
test_data = np.append(test_data, y_output.reshape(1,5), axis=0)
y_output_inv = scaler.inverse_transform(test_data)[-1][3]
# 重复上面的过程，直到得到所需天数的预测值
# 创建一个空列表用于存放未来几天的预测值
future_pred = []

# 设置预测天数为4（加上当前时间步共5天）
days = 4

# 循环进行预测并添加到列表中
for i in range(days):
    x_input = test_data[-n:]
    y_output = model.predict(x_input.reshape(1,n,5))
    test_data = np.append(test_data, y_output.reshape(1,5), axis=0)
    y_output_inv = scaler.inverse_transform(test_data)[-1][3]
    future_pred.append(y_output_inv)
# 打印出未来几天的预测值，并与真实值进行比较
print('The predicted closing prices for the next 5 days are:')
print(future_pred)

# 假设有真实数据可用
print('The actual closing prices for the next 5 days are:')
print('15.03 15.19 15.21 15.28')


