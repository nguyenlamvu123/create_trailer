import pandas as pd

df1 = pd.read_csv('items_list_1.csv')
df2 = pd.read_csv('items_list_2.csv')
df = pd.merge(
    df1, df2,
    on='item_id',
    how='outer',
    indicator=True,
##    left_index=True, right_index=True,
    ),
print(df1.to_string())
##print(len(df2.to_string()))
print(len(df1))
##print(len(df2))
##print(
##    len(
##        df
##        )
##    )
##print(df.items_name)
##print(df.to_string()) 
