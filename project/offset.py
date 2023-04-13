# import tushare as ts
# import pandas as pd
# import numpy as np
# import yfinance as yf
#
# # 设置参数
# stocks = ['600000.SH', '600519.SH', '000002.SZ', '000651.SZ']  # 股票代码
# money = 1000000  # 初始资金
# start_date = '2016-01-01'  # 回测起始时间
# end_date = '2022-03-28'  # 回测结束时间
# lookback = 20  # 布林带回看天数
# upperbound = 2  # 上轨线标准差倍数
# lowerbound = 2  # 下轨线标准差倍数
#
# # 获取历史数据
# data = pd.DataFrame()
# for stock in stocks:
#     stock_data = ts.get_k_data(stock, start=start_date, end=end_date, index=True)
#     stock_data.set_index('date', inplace=True)
#     stock_data['code'] = stock
#     data = pd.concat([data, stock_data], axis=0)
#
# # 计算布林带指标
# data['ma'] = data['close'].rolling(window=lookback).mean()
# data['std'] = data['close'].rolling(window=lookback).std()
# data['upper'] = data['ma'] + upperbound * data['std']
# data['lower'] = data['ma'] - lowerbound * data['std']
#
# # 获取港股市场数据
# hk_stock = yf.Ticker('^HSI')
# hk_data = hk_stock.history(start=start_date, end=end_date)[['Close']]
# hk_data.rename(columns={'Close': 'HK'}, inplace=True)
#
# # 合并数据
# data = pd.merge(data, hk_data, how='left', left_index=True, right_index=True)
#
# # 计算资产之间的相关系数
# returns = data.pct_change()
# cov = returns.cov()
# cor = cov / cov.values.diagonal().reshape(-1, 1) ** 0.5 / cov.values.diagonal() ** 0.5
#
# # 初始化投资组合
# portfolio = pd.DataFrame(index=data.index, columns=stocks + ['HK', 'cash'])
# portfolio.iloc[0] = np.append(np.zeros(len(stocks) + 1), money)
#
# # 回测交易策略
# for i in range(1, len(data)):
#     yesterday = data.index[i-1]
#     today = data.index[i]
#     weights = pd.Series(index=stocks + ['HK'], data=cor.loc[stocks + ['HK']].iloc[-1].values[:-1])
#     weights /= weights.sum()
#     weighted_returns = returns.loc[today][stocks + ['HK']] * weights
#     current_value = portfolio.iloc[i-1][:-1] * (1 + weighted_returns)
#     total_value = current_value.sum()
#     portfolio.loc[today]['cash'] = portfolio
import pandas as pd
import numpy as np
import tushare as ts
import statsmodels.api as sm
from datetime import datetime, timedelta

# 设置回测时间段
start_date = '20180101'
end_date = '20211231'

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
df_pct['Market'] = df_pct.mean(axis=1)

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
risk_premium = exp_return['Market'] - 0.03

# 计算每个资产的目标权重
target_weights = []
for i in range(len(stock_pool)):
    target_weight = weights_array[i]*(beta *df_pct.mean(axis=1)+ (1 - beta)*corr_matrix.iloc[i][stock_pool[i]])
    target_weights.append(target_weight)
target_weights /= np.sum(target_weights)
print(target_weight)

#获取每日大盘数据
df_index = pro.daily(ts_code='000001.SH', start_date=start_date, end_date=end_date)
df_index = df_index[['trade_date', 'close']].rename(columns={'close': 'Index'})
df_index['trade_date'] = pd.to_datetime(df_index['trade_date'], format='%Y%m%d')
df_index.set_index('trade_date', inplace=True)

# 合并资产和市场数据
df_all = pd.concat([df, df_index], axis=1)

# 计算动态资产权重和资产收益率
for i, date in enumerate(df_pct.index):
    if i == 0:
        dynamic_weights = np.array(list(weights.values()))
    else:
        exp_return = df_pct.loc[date].values * 252
        print(exp_return)
        std_dev = df_pct.loc[:date].std() * np.sqrt(252)
        y = exp_return - 0.03
        X = std_dev.values.reshape(-1, 1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]
        market_return = df_all.loc[date, 'Index'] / df_all.loc[:date, 'Index'].iloc[0] - 1
        risk_premium = market_return - 0.03
        print(risk_premium)
        target_weights = []
        for j in range(len(stock_pool)):
            target_weight = weights_array[j] * (beta * df_pct.mean(axis=1) + (1 - beta) * corr_matrix.iloc[j][stock_pool[j]])
            target_weights.append(target_weight)
        target_weights /= np.sum(target_weights)
        dynamic_weights = target_weights + risk_premium * np.linalg.inv(corr_matrix.values) @ (exp_return - target_weights)

# 计算组合收益率和波动率
portfolio_return = dynamic_weights @ df_pct.loc[date].values
portfolio_std_dev = np.sqrt(dynamic_weights @ corr_matrix.values @ dynamic_weights.T) * np.sqrt(252)
# 更新上一期的权重
prev_weights = dynamic_weights
# 计算交易成本
if i > 0:
    turnover = np.sum(np.abs(dynamic_weights - prev_weights))
    trading_cost = turnover * 0.001
else:
    trading_cost = 0

# 记录组合收益率、波动率和交易成本
df_all.loc[date, 'Portfolio Return'] = portfolio_return
df_all.loc[date, 'Portfolio Volatility'] = portfolio_std_dev
df_all.loc[date, 'Trading Cost'] = trading_cost

# 记录动态资产权重
for j, stock in enumerate(stock_pool):
    df_all.loc[date, f'{stock} Weight'] = dynamic_weights[j]

prev_weights = dynamic_weights

# 计算组合年化收益率、波动率和夏普比率
portfolio_return = df_all['Portfolio Return'].mean() * 252
portfolio_std_dev = df_all['Portfolio Volatility'].mean()
sharpe_ratio = portfolio_return / portfolio_std_dev

# 计算组合年化收益率和波动率
portfolio_annual_return = (1 + portfolio_return)**252 - 1
portfolio_annual_std_dev = portfolio_std_dev * np.sqrt(252)
# 输出结果
print(f'回测时间段：{start_date} 至 {end_date}')
print(f'资产池：{stock_pool}')
print(f'资产权重：{weights}')
print(f'资产相关系数矩阵：\n{corr_matrix}')
print(f'动态资产权重：\n{dynamic_weights}')
print(f'资产预期收益率：\n{exp_return}')
print(f'资产波动率：\n{std_dev}')
print(f'资本资产定价模型参数 beta：{beta}')
print(f'市场风险溢价：{risk_premium}')
print(f"每个资产的目标权重: {dict(zip(stock_pool, target_weights))}")
print(f"组合年化收益率: {portfolio_annual_return:.2%}")
print(f"组合年化波动率: {portfolio_annual_std_dev:.2%}")