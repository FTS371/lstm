import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time
# 设置股票代码和日期
code="600519"
stock_code = 'sh'+code
start_date = "20210103"
end_date = "20220101"

# 获取股票市场总貌
stock_sse_summary_df = ak.stock_sse_summary()
print(stock_sse_summary_df)

# 获取股票历史价格数据
stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
print(stock_zh_a_daily_qfq_df)

# 获取股票指标数据
stock_a_indicator_df = ak.stock_a_indicator_lg(symbol=code)
print(stock_a_indicator_df)

# 绘制市盈率的正方图和直方图
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].hist(stock_a_indicator_df.pe, bins=50)
ax[0].set_xlabel('PE')
ax[0].set_ylabel('Frequency')
ax[0].set_title('PE Histogram')
ax[1].hist(stock_a_indicator_df.pe_ttm, bins=50)
ax[1].set_xlabel('PE TTM')
ax[1].set_ylabel('Frequency')
ax[1].set_title('PE TTM Histogram')
plt.tight_layout()
plt.savefig('Data/'+code+'~1.png')

# 绘制市销率的曲线图
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stock_a_indicator_df.trade_date, stock_a_indicator_df.ps, c='b')
ax.set_xlabel('Date')
ax.set_ylabel('PS')
ax.set_title('PS Curve')
plt.savefig('Data/'+code+'~2.png')

# 绘制市盈率、市销率、市净率和市总值的图表
fig, axs = plt.subplots(nrows=4, figsize=(10, 10), sharex=True)
axs[0].plot(stock_a_indicator_df.trade_date, stock_a_indicator_df.pe, c='b')
axs[0].set_ylabel('PE')
axs[0].set_title('PE')

axs[1].plot(stock_a_indicator_df.trade_date, stock_a_indicator_df.ps, c='r')
axs[1].set_ylabel('PS')
axs[1].set_title('PS')

axs[2].plot(stock_a_indicator_df.trade_date, stock_a_indicator_df.pb, c='g')
axs[2].set_ylabel('PB')
axs[2].set_title('PB')

axs[3].plot(stock_a_indicator_df.trade_date, stock_a_indicator_df.total_mv, c='k')
axs[3].set_ylabel('Total MV')
axs[3].set_xlabel('Date')
axs[3].set_title('Total MV')
plt.tight_layout()
plt.savefig('Data/'+code+'~3.png')

def calculate_financial_indicators(stock_code):
    """计算指定股票的财务指标"""
    df = ak.stock_financial_analysis_indicator(symbol=stock_code)
    # 只计算过去五年的数据

    # 计算平均净资产收益率
    roe_mean = df['净资产收益率(%)'].replace('--', 0).astype(float).mean()
    indicator1 = roe_mean > 14

    # 计算市盈率
    lg_indicator_df = ak.stock_a_indicator_lg(symbol=stock_code)
    pe_mean = lg_indicator_df[lg_indicator_df.trade_date > datetime.date.today() - datetime.timedelta(days=30)].pe.mean()
    indicator2 = 0 < pe_mean < 30

    # 计算经营现金流
    cash_flow_per_share = float(df['每股经营性现金流(元)'].iat[1])
    indicator3 = cash_flow_per_share > 0

    # 计算净利润
    # 计算净利润
    net_profit_series = df['扣除非经常性损益后的净利润(元)']
    if '--' in net_profit_series.values:
        net_profit_this_year = 0.0
    else:
        net_profit_this_year = float(net_profit_series.iat[1])
    net_profit_last_five_years = net_profit_series.iloc[2:8].replace('--', 0).astype(float).max() / 10000
    indicator4 = net_profit_this_year > net_profit_last_five_years
    # 返回四个指标的结果
    return indicator1, indicator2, indicator3, indicator4


# 获取所有 A 股上市公司的股票代码和名称
all_stocks_df = ak.stock_zh_a_spot()
all_stocks_df = all_stocks_df[['代码', '名称']]

# 取前 300 只股票进行评估
top_300_stocks_df = all_stocks_df.head(100)
print(top_300_stocks_df)
# 计算所有股票的指标并评估综合财务状况
result_list = []
for index, row in top_300_stocks_df.iterrows():
    stock_code= row['代码'][2:]
    stock_name=row['名称']
    indicators = calculate_financial_indicators(stock_code)
    is_healthy = all(indicators)
    result_list.append({'股票代码': stock_code, '股票名称': stock_name, '指标1': indicators[0], '指标2': indicators[1], '指标3': indicators[2], '指标4': indicators[3], '综合评估': is_healthy})
    time.sleep(5)

# 将结果转换为 DataFrame 并打印
result_df = pd.DataFrame(result_list)
print(result_df)
# 保存结果到CSV文件
result_df.to_csv('output/financial_indicators.csv', index=False)
#
# # 可视化各个指标的结果
# import matplotlib.pyplot as plt
#
# # 统计指标1~4的数量
# counts = result_df.iloc[:, 2:6].sum()
#
# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.bar(counts.index, counts.values)
# ax.set_title('Financial Indicators')
# ax.set_xlabel('Indicators')
# ax.set_ylabel('Counts')
# plt.show()















# import datetime
# import time
# import numpy as np
# import pandas as pd
# import akshare as ak
#
#
# def filter_stocks(stock_list, indicators):
#     """
#     筛选股票
#
#     Parameters:
#     stock_list (pandas.DataFrame): 股票列表，包含股票代码和名称
#     indicators (dict): 股票筛选指标及其对应的参数
#
#     Returns:
#     pandas.DataFrame: 符合条件的股票列表，包含股票代码、名称和综合评估结果
#     """
#     # 创建结果 DataFrame
#     columns = ['code', 'name', '综合评估']
#     columns += [f'指标{i+1}' for i in range(len(indicators))]
#     result_df = pd.DataFrame(columns=columns)
#
#     # 遍历股票列表，筛选符合条件的股票
#     for idx, row in stock_list.iterrows():
#         code, name = row['code'], row['name']
#         indicators_result = []
#         for key, params in indicators.items():
#             # 计算指标结果
#             if key == '净资产收益率':
#                 indicators_result.append(calculate_return_on_equity(code, params))
#             elif key == '市盈率':
#                 indicators_result.append(calculate_pe_ratio(code, params))
#             elif key == '经营现金流':
#                 indicators_result.append(calculate_operating_cash_flow(code, params))
#             elif key == '净利润':
#                 indicators_result.append(calculate_net_profit(code, params))
#             else:
#                 raise ValueError(f'Unknown indicator: {key}')
#
#         # 计算综合评估结果
#         comprehensive_evaluation = all(indicators_result)
#
#         # 将结果添加到结果 DataFrame 中
#         result_df = result_df.append({
#             'code': code,
#             'name': name,
#             '综合评估': comprehensive_evaluation,
#             **{f'指标{i+1}': result for i, result in enumerate(indicators_result)}
#         }, ignore_index=True)
#
#         # 每处理完一个股票暂停 5 秒，防止请求频率过高
#         time.sleep(5)
#
#     return result_df
#
#
# def calculate_return_on_equity(stock_code, min_return_rate):
#     """
#     计算指定股票的净资产收益率是否高于指定的最小值
#
#     Parameters:
#     stock_code (str): 股票代码
#     min_return_rate (float): 最小净资产收益率
#
#     Returns:
#     bool: 净资产收益率是否高于指定的最小值
#     """
#     # 获取财务指标数据
#     financial_df = ak.stock_financial_analysis_indicator(stock_code)
#
#     # 过滤出近 5 年的净资产收益率
#     recent_return_df = financial_df.loc[financial_df.index > '2015-01-01', '净资产收益率(%)']
#
#     # 计算近 5 年的净资产收益率平均值是否高于指定的最小值
#     return_rate_mean = recent_return_df.replace('--', 0).astype(float).mean()
#     return return_rate_mean > min_return_rate
#
#
# def calculate_pe_ratio(stock_code, days):
#     """
#     计算指定天数内的市盈率均值
#     """
#     day = (datetime.datetime.now() - datetime.timedelta(days=days))
#     date_start = datetime.datetime(day.year, day.month, day.day, 0, 0, 0)  # 过去days天的数据
#     df = ak.stock_a_lg_indicator(stock=stock_code)
#     pe_mean = df[df.trade_date > date_start].pe.mean()
#     return pe_mean
#
#
# def calculate_net_profit_growth(stock_code, years):
#     """
#     计算指定年数内的净利润增长率
#     """
#     df = ak.stock_financial_analysis_indicator(stock=stock_code)  # 财务指标数据
#     net_profit = df['净利润(元)']
#     net_profit_growth = (net_profit.iloc[-1] / net_profit.iloc[-(years * 4)]) ** (1 / years / 4) - 1  # 每年4个财季
#     return net_profit_growth
#
#
# def calculate_operating_cash_flow_growth(stock_code, years):
#     """
#     计算指定年数内的经营现金流增长率
#     """
#     df = ak.stock_financial_analysis_indicator(stock=stock_code)  # 财务指标数据
#     operating_cash_flow = df['经营活动产生的现金流量净额(元)']
#     operating_cash_flow_growth = (operating_cash_flow.iloc[-1] / operating_cash_flow.iloc[-(years * 4)]) ** (
#             1 / years / 4) - 1  # 每年4个财季
#     return operating_cash_flow_growth
#
#
# def calculate_revenue_growth(stock_code, years):
#     """
#     计算指定年数内的营收增长率
#     """
#     df = ak.stock_financial_analysis_indicator(stock=stock_code)  # 财务指标数据
#     revenue = df['营业总收入(元)']
#     revenue_growth = (revenue.iloc[-1] / revenue.iloc[-(years * 4)]) ** (1 / years / 4) - 1  # 每年4个财季
#     return revenue_growth
#
#
# def calculate_comprehensive_evaluation(stock_code):
#     """
#     计算综合评估，综合考虑上述四个指标和其他因素
#     """
#     pe_mean_30d = calculate_pe_ratio(stock_code, 30)
#     pe_mean_365d = calculate_pe_ratio(stock_code, 365)
#     net_profit_growth_5y = calculate_net_profit_growth(stock_code, 5)
#     operating_cash_flow_growth_5y = calculate_operating_cash_flow_growth(stock_code, 5)
#     revenue_growth_5y = calculate_revenue_growth(stock_code, 5)
#
#     # 根据具体情况调整各项指标的权重
#     score = (pe_mean_30d + pe_mean_365d / 2) * 0.3 + net_profit_growth_5y * 0.3 + \
#             operating_cash_flow_growth_5y * 0.2 + revenue_growth_5y * 0.2
#
#     if score > 0:
#         return True
#     else:
#         return False



