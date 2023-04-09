import pandas as pd
import numpy as np
import tushare as ts
import statsmodels.api as sm
from datetime import datetime, timedelta

# 设置回测时间段
start_date = '2018-01-01'
end_date = '2021-12-31'

# 设置股票池
stock_pool = ['600036.SH', '601318.SH', '000858.SZ', '000002.SZ', '000651.SZ']

# 设置资产分配比例
weights = {'600036.SH': 0.2, '601318.SH': 0.2, '000858.SZ': 0.2, '000002.SZ': 0.2, '000651.SZ': 0.2}

# 设置资产相关系数矩阵
corr_matrix = pd.DataFrame({'600036.SH': [1.00, 0.65, 0.72, 0.59, 0.71],
                            '601318.SH': [0.65, 1.00, 0.67, 0.54, 0.61],
                            '000858.SZ': [0.72, 0.67, 1.00, 0.61, 0.69],
                            '000002.SZ': [0.59, 0.54, 0.61, 1.00, 0.55],
                            '000651.SZ': [0.71, 0.61, 0.69, 0.55, 1.00]},
                           index=['600036.SH', '601318.SH', '000858.SZ', '000002.SZ', '000651.SZ'])

# 获取行情数据
ts.set_token("928a7fbabe794c1e3312dc2117a8a4d597d4ff24bbe6ee475fa87726")
pro = ts.pro_api()
df = pd.DataFrame()
for stock in stock_pool:
    df_stock = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
    df_stock = df_stock[['trade_date', 'close']].rename(columns={'close': stock})
    df_stock['trade_date'] = pd.to_datetime(df_stock['trade_date'], format='%Y%m%d')
    df_stock.set_index('trade_date', inplace=True)
    df = pd.concat([df, df_stock], axis=1)

# 计算资产收益率和市场收益率
df_pct = df.pct_change().dropna()
df_pct['Market_return']=df_pct.mean(axis=1)
# 计算资产权重
weights_array = np.array(list(weights.values()))
weights_array /= np.sum(weights_array)
# 计算资产预期收益率和波动率
exp_return = df_pct.mean() * 252
std_dev = df_pct.std() * np.sqrt(252)
# 计算资本资产定价模型参数
y = exp_return.values - 0.03
X = std_dev.values.reshape(-1, 1)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
beta = model.params[1]
# 计算市场风险溢价
risk_premium = exp_return['Market_return'] - 0.03
# 计算每个资产的目标权重
target_weights = []
market_weight = 0.2 # 假设市场权重是0.2
for i in range(len(stock_pool)):
    target_weight = weights_array[i]*(beta * corr_matrix.iloc[i]['Market_return'] + (1 - beta)*corr_matrix.iloc[i][stock_pool[i]])
    target_weights.append(target_weight)
    target_weights /= np.sum(target_weights)

print(target_weights)