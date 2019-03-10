#! -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv("./data.csv", header=0)
print df.shape
columns = df.columns.values

new_columns = []
count =0
for col in columns:
    if col != 'is_click':
        new_columns.append('col_{0}'.format(count))
        count +=1
    else:
        new_columns.append('is_click')
df.columns = new_columns
ctr = np.random.rand(1,df.shape[0])[0]
print ctr
df["is_click"] = [1 if i>0.5 else -1 for i in ctr]

df[:100].to_csv("./train.csv",index=False)
df[-100:].to_csv("./test.csv", index=False)

