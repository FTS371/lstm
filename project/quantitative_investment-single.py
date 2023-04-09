import pandas as pd
import numpy as np
import yfinance as yf
import talib


def get_stock_data(symbol, start_date, end_date):
    # 获取股票数据
    stock_data = yf.download(symbol, start_date, end_date)
    stock_data = pd.DataFrame(stock_data['Adj Close'])
    stock_data = stock_data.dropna()
    return stock_data


def strategy(stock_data):
    # 计算指标
    stock_data['sma_20'] = talib.SMA(stock_data['Adj Close'], timeperiod=20)
    stock_data['sma_50'] = talib.SMA(stock_data['Adj Close'], timeperiod=50)
    stock_data['rsi'] = talib.RSI(stock_data['Adj Close'], timeperiod=14)
    stock_data['atr'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Adj Close'], timeperiod=14)
    stock_data['upper_band'], stock_data['middle_band'], stock_data['lower_band'] = talib.BBANDS(
        stock_data['Adj Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 信号：买入条件
    buy_signal = ((stock_data['Adj Close'] > stock_data['sma_20']) & (stock_data['Adj Close'] > stock_data['sma_50']) &
                  (stock_data['rsi'] < 30) & (stock_data['Adj Close'] < stock_data['lower_band']))

    # 信号：卖出条件
    sell_signal = ((stock_data['Adj Close'] < stock_data['sma_20']) | (stock_data['Adj Close'] < stock_data['sma_50']) |
                   (stock_data['rsi'] > 70) | (stock_data['Adj Close'] > stock_data['upper_band']))

    # 计算止损价格
    stock_data['stop_loss'] = stock_data['Adj Close'] - 2 * stock_data['atr']

    # 初始化持仓
    positions = pd.Series(data=np.nan, index=stock_data.index)

    # 计算信号和持仓
    for i in range(1, len(stock_data)):
        # 如果没有持仓且出现买入信号，则建立新仓位
        if np.isnan(positions[i - 1]) and buy_signal[i]:
            positions[i] = 1
        # 如果持有仓位且出现卖出信号，则清空仓位
        elif positions[i - 1] == 1 and sell_signal[i]:
            positions[i] = np.nan
        # 否则，保持持仓不变
        else:
            positions[i] = positions[i - 1]

        # 如果当前持有仓位且价格低于止损价，则清空仓位
        if positions[i] == 1 and stock_data['Adj Close'][i] < stock_data['stop_loss'][i]:
            positions[i] = np.nan

        # 计算每日收益
    daily_returns = (stock_data['Adj Close'].pct_change() * positions.shift(1)).dropna()

    # 计算策略收益
    strategy_returns = (daily_returns + 1).cumprod()

    return strategy_returns


