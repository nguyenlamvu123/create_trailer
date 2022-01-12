import os
import matplotlib.pyplot as plt
import numpy as np

class Howk_ex7():
    """7.1. Exercise 1:

    Randomly choose 3 shops, use one line plot figure to show the total turnover of them each month over 33 months (from January 2013 is 0, February 2013 is 1,..., October 2015 is 33.)
    7.2. Exercise 2:

    Randomly choose 1 item, use one combo figure with bar plot and line plot to show the sales and the turnover of them over 33 months (from January 2013 is 0, February 2013 is 1,..., October 2015 is 33.)
    7.3. Exercise 3:

    Randomly choose 3 items, use one bar plot figure to show the sales of them in 3 years 2013, 2014, and 2015.
    7.4. Exercise 4:

    Randomly choose 1 shop, use stack plot figure to show the contribution of turnover of each item each month over 33 months (from January 2013 is 0, February 2013 is 1,..., October 2015 is 33.)
    7.5. Exercise 5:

    Randomly choose 1 shop, use pie plot figure to show the contribution of turnover (in percent) of each item in 3 years 2013, 2014, and 2015. (1 pie plot figure for each year, use subplots to put all 3 figures together)
    7.6. Exercise 6:

    Randomly choose 1 item, use histogram figure to show the distribution of sales of the item over 33 months (from January 2013 is 0, February 2013 is 1,..., October 2015 is 33.)"""
    def __init__(self):
        pass

    def csssv(self, fillle = 'sales.csv'): #def csssv(fillle = 'sales.csv'): #
        import pandas as pd
        samgiongzon = os.path.join(
            os.getcwd(),
            'predict_future_sales',
            )
        df = pd.read_csv(
            os.path.join(
                samgiongzon,
                fillle,
                )
            )
##        print(df)
        return df#

    def crtdata(self):
        import pandas as pd
        pd.pandas.set_option('display.max_columns', None)
        from homwk_saleshopcsv import itemsss_df
        sales_df = self.csssv()#df1 = csssv('shops.csv')#
##        df21 = self.csssv()#df2 = csssv()#
##        df22 = self.csssv()#df2 = csssv()#
##        items_df = pd.merge(df21, df22, how='outer', on='shop_id'))
        items_df = itemsss_df()
        sales_item_name_df = pd.merge(sales_df, items_df, on='item_id', how='left')
##################################total turnover, year 
        sales_item_name_df['total_turnover'] = sales_item_name_df.item_cnt_day*sales_item_name_df.item_price
        sales_item_name_df['year'] = sales_item_name_df.date.apply(
            lambda x: str(x.split('.')[-1])
            )
##################################total turnover, year
        sales_item_name_df.date = sales_item_name_df.date.apply(
            lambda x: str(x.split('.')[-1])+ '-'+ str(x.split('.')[-2])
            )
##        dattta = sales_item_name_df[sales_item_name_df.shop_id==25].groupby(
##            by=['date']
##            ).agg({'total_turnover': sum})
##        dattta = dattta.set_index("date", drop=False, inplace=True)
##        print(sales_item_name_df)#.date)#(dattta)#
##        return self.Exx1(dattta)#.total_turnover)
        return sales_item_name_df, sales_item_name_df.shop_id.unique()

    def titlelabellegent(
        self,
        leg=None,
        xl=None,
        yl=None,
        tit=None,
        ):
        plt.xticks(fontsize=10, rotation=90)

        if leg!=None: plt.legend(loc=leg)
        plt.xlabel(xl)
        plt.ylabel(yl)
        if tit!=1: plt.title(tit)

    def Exx1(self, sales_item_name_df, danhsach):
        for i in range(3):
            shop___id = np.random.choice(danhsach)
            plt.subplot(3, 1, i+1).set_title(f'shop_id = {shop___id}')
            dattta = sales_item_name_df[sales_item_name_df.shop_id==shop___id].groupby(
                by=['date']
                ).agg({'total_turnover': sum})
            plt.plot(dattta, label='ex1')#plt.plot(x, label='ex1')
            self.titlelabellegent(
                xl='date',
                yl='total_turnover',
                tit=1,
                )
            np.delete(danhsach, shop___id)
##            self.titlelabellegent()
            
        plt.show()
        
    def Exx2(self, sales_item_name_df):#, danhsach):
        danhsach = sales_item_name_df.item_id.unique()
        shop___id = np.random.choice(danhsach)
        dattta = sales_item_name_df[sales_item_name_df.item_id==shop___id].groupby(
            by=['date']
            ).agg({'total_turnover': sum, 'item_cnt_day': sum})
        
##        plt.plot(dattta.index, dattta['total_turnover'], label='ex2')
##        plt.bar(dattta.index, dattta['item_cnt_day'], label='ex2')#error 

        self.titlelabellegent(
            xl='date',
##            yl='total_turnover',
            tit=f'item_id = {shop___id}',
            )
        
        ax = dattta['total_turnover'].plot(kind='bar')
        ax.set_ylabel("total_turnover",fontsize=10, rotation=90)
        a1x = dattta['item_cnt_day'].plot(style='o--', c='black', secondary_y=True)
        a1x.set_ylabel("item_cnt_day",fontsize=10, rotation=90)
##https://stackoverflow.com/questions/68738683/combining-a-bar-plot-and-a-line-plot-in-matplotlib-without-shifting-the-bar-plot
        
        plt.show()

    def Exx3(self, sales_item_name_df):#, danhsach):
        danhsach = sales_item_name_df.item_id.unique()
        shop___id = np.random.choice(danhsach)
        dattta = sales_item_name_df[sales_item_name_df.item_id==shop___id].groupby(
            by=['year']
            ).agg({'item_cnt_day': sum})
##        print(sales_item_name_df[sales_item_name_df.item_id==9880])
        index = np.arange(3)
        dattta['item_cnt_day'].plot(kind='bar')
        self.titlelabellegent(
            xl='year',
            yl='item_cnt_day',
            tit=f'item_id = {shop___id}',
            )
        plt.show()

    def Exx4(self, sales_item_name_df, danhsach):#):#
        shop___id = np.random.choice(danhsach)
        dattta = sales_item_name_df[sales_item_name_df.item_id==shop___id].groupby(
            by=['date', 'item_id']
            ).agg({'total_turnover': sum})
##################################################################################################################################################        plt.stackplot(category, value_1, value_2, value_3, colors=['r', 'g', 'b'], label='ex1')
        self.titlelabellegent();plt.show()

    def Exx5(self):
        plt.pie(value, labels=category, startangle=90, label='ex1')
        self.titlelabellegent();plt.show()

    def Exx6(self):
        plt.hist(y,                 # data
                 bins=50,           # number of bar        
                 alpha=0.5,         # opacity
                 color='green',     # bar color
                 edgecolor='black', # edge of bar color
                 label='ex1') 
        self.titlelabellegent();plt.show()
##Howk_ex7().df1
##Howk_ex7().df2
samgiongzon = Howk_ex7()
sales_item_name_df, danhsach = samgiongzon.crtdata()
samgiongzon.Exx4(sales_item_name_df, danhsach)#samgiongzon.Exx3(sales_item_name_df)#
##samgiongzon.Exx3(sales_item_name_df)#samgiongzon.Exx2(sales_item_name_df, danhsach)
##samgiongzon.Exx2(sales_item_name_df)#samgiongzon.Exx2(sales_item_name_df, danhsach)
##samgiongzon.Exx1(sales_item_name_df, danhsach)

import datetime
x = datetime.datetime.now();print(x)
if x > datetime.datetime(2022, 1, 12, 19, 00, 00):
    input('commit to git!!!')
    import shutil
    shutil.rmtree(r'/media/asrock/New Volume/VNPhatLoc/create_trailer/')
