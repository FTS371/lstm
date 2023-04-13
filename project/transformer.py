
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * div)
        self.pe[:, 1::2] = torch.cos(pos * div)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # linearly project queries, keys and values
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        # split each query, key and value into num_heads
        Q = Q.view(batch_size * self.num_heads, -1, self.head_dim)
        K = K.view(batch_size * self.num_heads, -1, self.head_dim)
        V = V.view(batch_size * self.num_heads, -1, self.head_dim)

        # compute dot product of query with key for each head
        attention_scores = torch.bmm(Q, K.transpose(1, 2))
        attention_scores = attention_scores / np.sqrt(self.d_model)

        # apply mask (if specified)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # apply softmax to obtain weights for each head
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        # apply dropout to attention weights
        attention_weights = nn.functional.dropout(attention_weights, p=0.1)

        # obtain the weighted sum of values for each head
        attention_output = torch.bmm(attention_weights, V)

        # concatenate attention outputs from all heads and apply final linear layer
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        attention_output = self.Wo(attention_output)

        return attention_output, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # Multi-head attention
        attn_output, _ = self.multihead_attention(x, x, x, mask)
        # Layer normalization and residual connection
        x = self.norm1(x + self.dropout1(attn_output))
        # Feedforward network
        ff_output = self.feedforward(x)
        # Layer normalization and residual connection
        x = self.norm2(x + self.dropout2(ff_output))
        return x


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
num_decoder_layers = 0.5

model = TransformerBlock(n_features, nhead, num_encoder_layers, num_decoder_layers)
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


# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
#
# # 加载数据集
# df = pd.read_csv('Data/Shanghai(20230316).csv')
# data = df['close'].values
#
# # 划分训练集和测试集
# seq_len = 30
# split = int(0.8 * len(data))
# train_data = data[:split]
# test_data = data[split - seq_len:]
#
#
# # 准备数据
# def prepare_data(data, seq_len):
#     X = []
#     Y = []
#     for i in range(seq_len, len(data)):
#         X.append(data[i - seq_len:i])
#         Y.append(data[i])
#     return np.array(X), np.array(Y)
# X_train, Y_train = prepare_data(train_data, seq_len)
# X_test, Y_test = prepare_data(test_data, seq_len)
#
#
# # 定义Transformer模型
# class TransformerModel(nn.Module):
#     def __init__(self, input_size, output_size, d_model, nhead, num_layers, dim_feedforward):
#         super().__init__()
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
#             num_layers=num_layers
#         )
#         self.linear = nn.Linear(input_size, d_model)
#         self.output = nn.Linear(d_model, output_size)
#
#     def forward(self, x):
#         x = self.linear(x)
#         x = x.permute(1, 0, 2)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)
#         x = self.output(x[:, -1, :])
#         return x
#
#
# # 训练模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = TransformerModel(seq_len, 1, d_model=128, nhead=8, num_layers=6, dim_feedforward=512).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
# epochs = 100
# batch_size = 64
# num_batches = len(X_train) // batch_size
# train_losses = []
#
# for epoch in range(epochs):
#     for i in range(num_batches):
#         start = i * batch_size
#         end = start + batch_size
#         x_batch = X_train[start:end]
#         y_batch = Y_train[start:end]
#         x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
#         y_batch = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1).to(device)
#         optimizer.zero_grad()
#         y_pred = model(x_batch)
#         loss = criterion(y_pred, y_batch)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())
#
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, loss: {loss.item()}')
#
# # 测试模型
# def test(model, x_test, y_test):
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(torch.tensor(x_test, dtype=torch.float32).to(device)).cpu().numpy()
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     corr_coef = np.corrcoef(y_test, y_pred)[0][1]
#     print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Correlation Coefficient: {corr_coef:.4f}")
#
# # # 绘制预测图表
# # plt.figure(figsize=(10, 6))
# # plt.plot(y_test, label='True')
# # plt.plot(y_pred, label='Prediction')
# # plt.legend()
# # plt.show()

