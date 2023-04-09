import tushare as ts
import pandas as pd
# 登录tushare账号，获取token
ts.set_token("")
pro = ts.pro_api()

# 设置股票池和起止日期
stocks = ["600519.SH", "000858.SZ", "000333.SZ", "601318.SH", "000651.SZ"]
start_date = "20191203"
end_date = "20220101"

# 获取股票池的历史数据
data = {}
for stock in stocks:
    stock_data = pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
    stock_data.set_index("trade_date", inplace=True)
    data[stock] = stock_data
print(stock_data)
# 计算布林带指标
for stock in stocks:
    data[stock]["MA20"] = data[stock]["close"].rolling(20).mean()
    data[stock]["std"] = data[stock]["close"].rolling(20).std()
    data[stock]["upper"] = data[stock]["MA20"] + 2 * data[stock]["std"]
    data[stock]["lower"] = data[stock]["MA20"] - 2 * data[stock]["std"]

# 进行交易
cash = 100000  # 初始资金
position = {}  # 持仓
buy_list = []  # 买入列表
sell_list = []  # 卖出列表
for i, date in enumerate(data[stocks[0]].index):
    for stock in stocks:
        # 如果当前价格突破上轨线，则买入该股票
        if data[stock]["close"][date] > data[stock]["upper"][date]:
            buy_list.append(stock)
        # 如果当前价格跌破下轨线，则卖出该股票
        elif stock in position and data[stock]["close"][date] < data[stock]["lower"][date]:
            sell_list.append(stock)
        # 如果当前价格在上下轨之间，则不做任何操作
        else:
            pass

    # 先处理卖出订单
    for stock in sell_list:
        if stock in position:
            cash += data[stock]["close"][date] * position[stock]
            position.pop(stock)

    # 再处理买入订单
    buy_list = list(set(buy_list))  # 去重
    for stock in buy_list:
        if cash >= 0 and cash >= data[stock]["close"][date]:
            shares = cash // data[stock]["close"][date]
            position[stock] = shares
            cash -= data[stock]["close"][date] * shares

    # 清空买卖列表
    buy_list = []
    sell_list = []

    # 输出当前资产情况
    if i % 30 == 0:
        total_value = cash + sum([data[stock]["close"][date] * position[stock] for stock in position])
        print("Date: %s, Cash: %f, Total Value: %f" % (date, cash, total_value))
