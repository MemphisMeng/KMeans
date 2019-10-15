import pandas as pd

df1 = pd.read_csv('output.csv')
df2 = pd.read_csv('output2.csv')

df1 = df1.merge(df2, left_index=True, right_index=True, how='right')
d = 0
for index_1, row_1 in df1.iterrows():
    for index_2, row_2 in df1.iterrows():
        if (row_1['label_x'] == row_2['label_x'] and row_1['label_y'] != row_2['label_y']) or \
                (row_1['label_y'] == row_2['label_y'] and row_1['label_x'] != row_2['label_x']):
            d += 1
print(d)
