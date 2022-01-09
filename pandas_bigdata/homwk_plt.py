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
        print(df)
        return df#

    def crtdata(self):
        import pandas as pd
        df1 = self.csssv('shops.csv')#df1 = csssv('shops.csv')#
        df2 = self.csssv()#df2 = csssv()#
        print(pd.merge(df1, df2, how='outer', on='shop_id'))

    def titlelabellegent():
        plt.legend(loc='upper left')
        plt.xlabel("Category")
        plt.ylabel("Value")
        plt.title("This is my figure")

    def Exx1(self):
        plt.plot(x, np.sin(x), label='ex1')
        titlelabellegent();plt.show()
        
    def Exx2(self):
        plt.plot(x, np.sin(x), label='ex1')
        plt.bar(index, value_1, label='ex1')
        titlelabellegent();plt.show()

    def Exx3(self):
        index = np.arange(5)
        plt.bar(index-0.2, value_1, width=0.2, label='ex1')
        plt.bar(index, value_2, width=0.2, label='ex1')
        plt.bar(index+0.2, value_3, width=0.2, label='ex1')
        titlelabellegent();plt.show()

    def Exx4(self):
        plt.stackplot(category, value_1, value_2, value_3, colors=['r', 'g', 'b'], label='ex1')
        titlelabellegent();plt.show()

    def Exx5(self):
        plt.pie(value, labels=category, startangle=90, label='ex1')
        titlelabellegent();plt.show()

    def Exx6(self):
        plt.hist(y,                 # data
                 bins=50,           # number of bar        
                 alpha=0.5,         # opacity
                 color='green',     # bar color
                 edgecolor='black', # edge of bar color
                 label='ex1') 
        titlelabellegent();plt.show()
##Howk_ex7().df1
##Howk_ex7().df2
##Howk_ex7().crtdata()

##import datetime
##x = datetime.datetime.now();print(x)
##if x>datetime.datetime(2022, 1, 7, 19, 00):
##    input('commit to git!!!')
##    import shutil
##    shutil.rmtree(r'/media/asrock/New Volume/VNPhatLoc/VuIbcCrawler/07012022/')
