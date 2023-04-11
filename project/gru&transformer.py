import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# 1. 数据准备
df = pd.read_csv('Data/Shanghai(20230316).csv')
train_size = int(len(df) * 0.7)
train_data = df[:train_size].copy()
test_data = df[train_size:].copy()
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 2. 数据生成器
n_input = 30
n_features = 5
train_generator = TimeseriesGenerator(train_data, train_data[:, 1], length=n_input, batch_size=32)
test_generator = TimeseriesGenerator(test_data, test_data[:, 1], length=n_input, batch_size=1)

# 3. GRU模型搭建
gru_model = Sequential()
gru_model.add(GRU(128, input_shape=(n_input, n_features), return_sequences=True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(64))
gru_model.add(Dense(1))
gru_model.compile(optimizer='adam', loss='mse')

# 4. GRU模型训练
early_stop = EarlyStopping(monitor='val_loss', patience=5)
gru_model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stop])

# 5. GRU模型预测
y_pred_gru = gru_model.predict(test_generator)
y_pred_gru = scaler.inverse_transform(y_pred_gru)
y_test = scaler.inverse_transform(test_data[n_input:, 1].reshape(-1, 1))

# 6. Transformer模型搭建
def transformer_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    embedding_layer = Embedding(100, 64)(inputs)
    attention_layer = MultiHeadAttention(num_heads=8, key_dim=64)(embedding_layer, embedding_layer)
    attention_layer = LayerNormalization()(attention_layer)
    attention_layer = Dropout(0.2)(attention_layer)
    gru_layer = GRU(32)(attention_layer)
    outputs = Dense(1)(gru_layer)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mse')
    return model

transformer_model = transformer_model((n_input, n_features))

# 7. Transformer模型训练
early_stop = EarlyStopping(monitor='val_loss', patience=5)
transformer_model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stop])

# 8. Transformer模型预测
y_pred_transformer = transformer_model.predict(test_generator)
y_pred_transformer = scaler.inverse_transform(y_pred_transformer)
# 9. 计算指标
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return mse, rmse, mae, r2, corr

mse_cnn, rmse_cnn, mae_cnn, r2_cnn, corr_cnn = evaluate(y_test[0], y_pred_cnn[:, 0])
mse_rnn, rmse_rnn, mae_rnn, r2_rnn, corr_rnn = evaluate(y_test[0], y_pred_rnn[:, 0])
mse_transformer, rmse_transformer, mae_transformer, r2_transformer, corr_transformer = evaluate(y_test[0], y_pred_transformer[:, 0])

print(f"CNN模型：MSE={mse_cnn:.4f}, RMSE={rmse_cnn:.4f}, MAE={mae_cnn:.4f}, R2={r2_cnn:.4f}, Corr={corr_cnn:.4f}")
print(f"RNN模型：MSE={mse_rnn:.4f}, RMSE={rmse_rnn:.4f}, MAE={mae_rnn:.4f}, R2={r2_rnn:.4f}, Corr={corr_rnn:.4f}")
print(f"Transformer模型：MSE={mse_transformer:.4f}, RMSE={rmse_transformer:.4f}, MAE={mae_transformer:.4f}, R2={r2_transformer:.4f}, Corr={corr_transformer:.4f}")

# 10. 绘制预测图表
plt.figure(figsize=(16, 8))
plt.plot(y_test[0], label='True')
plt.plot(y_pred_cnn[:, 0], label='CNN Predicted')
plt.legend()
plt.title('CNN Stock Price Prediction')
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(y_test[0], label='True')
plt.plot(y_pred_rnn[:, 0], label='RNN Predicted')
plt.legend()
plt.title('RNN Stock Price Prediction')
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(y_test[0], label='True')
plt.plot(y_pred_transformer[:, 0], label='Transformer Predicted')
plt.legend()
plt.title('Transformer Stock Price Prediction')
plt.show()

