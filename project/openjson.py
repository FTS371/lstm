# import json
# with open('query.json') as f:
#   data = json.load(f)
#   print(data)
import pandas as pd
df = pd.read_json(r'/Data/dahua.json')
df.to_csv(r'E:/diploma-project/project/dahua.csv', index=None)