# import tushare as ts
# ts.set_token('')
# pro = ts.pro_api()
# df = pro.daily(ts_code='002236.SZ', start_date='20180701', end_date='20230301')
# print(df) //Tushare
import requests
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=002236.SHZ&apikey=EUV62MW1XVFCIS7T'
r = requests.get(url)
data = r.json()

print(data)