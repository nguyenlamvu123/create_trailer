import pandas as pd
import numpy as np 
##############################################################################################################
def process_df(_df):
    _df = _df.dropna(axis=1).drop(columns='_merge')
    
    rename_dict = dict()
    for col_name in _df.columns:
        if '_x' in col_name or '_y' in col_name:
            rename_dict[col_name] = col_name[:-2]
    
    new_df = _df.rename(columns=rename_dict)
    return new_df
item_list_1_df = pd.read_csv('./predict_future_sales/items_list_1.csv')
item_list_2_df = pd.read_csv('./predict_future_sales/items_list_2.csv')
item_categories_df = pd.read_csv('./predict_future_sales/item_categories.csv')
sales_df = pd.read_csv('./predict_future_sales/sales.csv')

def itemsss_df(item_list_1_df=item_list_1_df, item_list_2_df=item_list_2_df):
    merged_df = pd.merge(item_list_1_df, item_list_2_df, on='item_id', how='outer', indicator=True)
    item_only_list_1_df = merged_df[merged_df._merge == 'left_only'];item_only_list_1_df = process_df(item_only_list_1_df)
    item_only_list_2_df = merged_df[merged_df._merge == 'right_only'];item_only_list_2_df = process_df(item_only_list_2_df)
    item_both_list_df = merged_df[merged_df._merge == 'both'];item_both_list_df = process_df(item_both_list_df)
    item_both_list_df = item_both_list_df.loc[:,~item_both_list_df.columns.duplicated()]
##    items_df = pd.concat([item_only_list_1_df, item_only_list_2_df, item_both_list_df])#;items_df.to_csv('items.csv', index=False)
    return pd.concat([item_only_list_1_df, item_only_list_2_df, item_both_list_df])#;items_df.to_csv('items.csv', index=False)
if __name__ == '__main__':
    items_df = itemsss_df(item_list_1_df, item_list_2_df)

    print(f'There are {len(item_list_1_df)} items in list 1')
    print(f'There are {len(item_list_2_df)} items in list 2')
##    print(f'There are {len(item_only_list_1_df)} items in only list 1')
##    print(f'There are {len(item_only_list_2_df)} items in only list 2')
##    print(f'There are {len(item_both_list_df)} items in both list 1 and list 2')
    print(f'There are {len(item_categories_df)} categories in the dataset')
    print(f'There are {len(items_df)} items in items.csv')
##############################################################################################################
def check_digits(name):
    for char in name:
        if char.isdigit():
            return True
    return False
def check_fifa(name):
    if 'FIFA' in name.upper():
        return True
    return False
if __name__ == '__main__':
    items_df['is_digits_in_name'] = items_df.item_name.apply(check_digits);item_name_with_digits_df = items_df.loc[items_df.is_digits_in_name]
    items_df['is_fifa_in_name'] = items_df.item_name.apply(check_fifa);item_name_with_fifa_df = items_df.loc[items_df.is_fifa_in_name]

    print(f'There are {len(item_name_with_digits_df)} items with digits in item_name')
    print(f'There are {len(item_name_with_fifa_df)} items with FIFA in item_name')
##############################################################################################################
if __name__ == '__main__':
    merged_item_cat_df = pd.merge(items_df, item_categories_df, how='left', on='item_category_id')

    grouped_merged_item_cat_df = merged_item_cat_df.groupby(
        by=['item_category_id', 'item_category_name']
        ).agg(
            {'item_id': list}
            ).reset_index()
    grouped_merged_item_cat_df['num_of_items'] = grouped_merged_item_cat_df.item_id.apply(len)

    max_num_items = np.max(grouped_merged_item_cat_df.num_of_items)
    max_num_of_items_df = grouped_merged_item_cat_df.loc[
        grouped_merged_item_cat_df.num_of_items == max_num_items]
    min_num_items = np.min(grouped_merged_item_cat_df.num_of_items)
    min_num_of_items_df = grouped_merged_item_cat_df.loc[
        grouped_merged_item_cat_df.num_of_items == min_num_items]
##############################################################################################################
def find_highest_lowest_price(df, year):
    sales_year_df = df.loc[df.date == year]

    highest_df = sales_year_df.loc[sales_year_df.item_price == np.max(sales_year_df.item_price)]
    lowest_df = sales_year_df.loc[sales_year_df.item_price == np.min(sales_year_df.item_price)]

    average_price = np.average(sales_year_df.item_price)
    return highest_df, lowest_df, average_price
def find_highest_lowest_sales(df, year):
    sales_year_df = df.loc[df.date == year]

    grouped_df = sales_year_df.groupby(by='item_id').agg({'item_cnt_day': sum}).reset_index()

    highest_df = grouped_df.loc[grouped_df.item_cnt_day == np.max(grouped_df.item_cnt_day)]
    lowest_df = grouped_df.loc[grouped_df.item_cnt_day == np.min(grouped_df.item_cnt_day)]

    average_sales = np.average(grouped_df.item_cnt_day)
    return highest_df, lowest_df, average_sales

if __name__ == '__main__':
    sales_item_name_df = pd.merge(sales_df, items_df, on='item_id', how='left')###########################

    sales_item_name_df.date = sales_item_name_df.date.apply(lambda x: x.split('.')[-1])

    year_list = sales_item_name_df.date.unique()
    for year in year_list:
        print('year', year)
        highest_df, lowest_df, average_price = find_highest_lowest_price(sales_item_name_df, year)
        print('highest_df \n', highest_df)
        print('lowest_df \n', lowest_df)
        print('average_price \n', average_price)
    for year in year_list:
        print(year)
        highest_df, lowest_df, average_sales = find_highest_lowest_sales(sales_item_name_df, year)
        print('highest_df \n', highest_df)
        print('lowest_df \n', lowest_df)
        print('average_sales \n', average_sales)
