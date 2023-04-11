import requests
import pandas as pd
from bs4 import BeautifulSoup

# 设置股票代码
stock_code = '600519'

# 获取股票所属行业
def get_industry(stock_code):
    url = 'https://gupiao.baidu.com/stock/' + stock_code + '.html'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    industry = soup.find('a', {'class': 'c-gray', 'href': '/s?from=pc&name=s_' + stock_code + '&tb=industry'}).text
    return industry

# 获取股票财务报表数据
def get_financials(stock_code):
    url = 'http://quotes.money.163.com/f10/zycwzb_' + stock_code + '.html'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'table_bg001'})
    headers = table.find_all('th')
    data = []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        if len(cols) == len(headers):
            data.append(cols)
    df = pd.DataFrame(data[1:], columns=headers)
    return df

# 计算市盈率
def calculate_pe(stock_code):
    df = get_financials(stock_code)
    eps = df.loc[df['报表日期'] == '基本每股收益（元）']['2019.12'].values[0]
    price = float(requests.get('http://hq.sinajs.cn/list=sh' + stock_code).text.split(',')[3])
    pe = price / eps
    return pe

# 计算市净率
def calculate_pb(stock_code):
    df = get_financials(stock_code)
    bvps = df.loc[df['报表日期'] == '每股净资产（元）']['2019.12'].values[0]
    price = float(requests.get('http://hq.sinajs.cn/list=sh' + stock_code).text.split(',')[3])
    pb = price / bvps
    return pb

# 获取股票所属行业
industry = get_industry(stock_code)
print('行业：', industry)

# 计算市盈率
pe = calculate_pe(stock_code)
print('市盈率：', pe)

# 计算市净率
pb = calculate_pb(stock_code)
print('市净率：', pb)
