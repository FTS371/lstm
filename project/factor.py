import pandas as pd
import numpy as np

# 导入数据
stock_data = pd.read_csv('Data/shanghai.csv')

# 计算因子
stock_data['Factor1'] = ...
stock_data['Factor2'] = ...
stock_data['Factor3'] = ...

# 标准化因子
stock_data['Factor1'] = (stock_data['Factor1'] - stock_data['Factor1'].mean()) / stock_data['Factor1'].std()
stock_data['Factor2'] = (stock_data['Factor2'] - stock_data['Factor2'].mean()) / stock_data['Factor2'].std()
stock_data['Factor3'] = (stock_data['Factor3'] - stock_data['Factor3'].mean()) / stock_data['Factor3'].std()

# 计算综合得分
stock_data['Score'] = stock_data['Factor1'] * 0.3 + stock_data['Factor2'] * 0.3 + stock_data['Factor3'] * 0.4

# 按得分排序
stock_data = stock_data.sort_values(by='Score', ascending=False)

# 选取前10只股票
selected_stocks = stock_data.head(10)['StockCode'].tolist()
