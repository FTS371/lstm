import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, n_features, nhead, num_encoder_layers, num_decoder_layers):
        super().__init__()
        embed_dim = nhead * 64
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)

        self.decoder = nn.Linear(n_features, 1)

    def forward(self, src, tgt):
        # src shape: (seq_len, batch, n_features)
        # tgt shape: (seq_len, batch, n_features)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        output = self.transformer(src, tgt)
        output = self.decoder(output)
        return output

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

# 转换为张量
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# 调整维度为 (seq_len, batch, n_features)
X_train = X_train.unsqueeze(1)
y_train = y_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
y_test = y_test.unsqueeze(1)

# 定义模型和优化器
n_features = 1
nhead = 2
num_encoder_layers = 2
num_decoder_layers = 2
model = TransformerModel(n_features, nhead, num_encoder_layers, num_decoder_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

# 训练模型
train_losses = []
test_losses = []
for epoch in range(epochs):
    # 训练集上训练
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train, y_train[:-1])
    loss = criterion(y_pred, y_train[1:])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 测试集上验证
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test, y_test[:-1])
        test_loss = criterion(y_test_pred, y_test[1:])
        test_losses.append(test_loss.item())

    # 打印训练结果
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 计算预测结果
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train, y_train[:-1])
    y_test_pred = model(X_test, y_test[:-2])  # 修改这里

# 反归一化
y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
y_train = scaler.inverse_transform(y_train[1:].reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test[1:-1].reshape(-1, 1)).flatten()


# 计算预测误差
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_corr = np.corrcoef(y_train, y_train_pred)[0, 1]
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_corr = np.corrcoef(y_test, y_test_pred)[0, 1]

# 输出预测误差和决定系数
print(f'Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}, Train R2: {train_r2:.2f}, Train Corr: {train_corr:.2f}')
print(f'Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, Test R2: {test_r2:.2f}, Test Corr: {test_corr:.2f}')

# 绘制股票预测图表
plt.figure(figsize=(12, 6))
plt.plot(df_train.index, y_train, label='Training Data')
plt.plot(df_test.index, y_test, label='Test Data')
plt.plot(df_train.index[1:], y_train_pred, label='Training Prediction')
plt.plot(df_test.index[1:], y_test_pred, label='Test Prediction')
plt.legend()
plt.title('Stock Closing Price Prediction with Transformer Model')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()