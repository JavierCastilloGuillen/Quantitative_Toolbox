import pandas as pd
import math
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
sns.set_theme()


# Dates
today = datetime.datetime.today()
to_date = today.strftime('%d/%m/%Y')
days_back = datetime.timedelta(days=365)
date_ago = today - days_back
from_date = date_ago.strftime('%d/%m/%Y')

# Data Retreival
def get_sp500_tickers():

    """Retrieves S&P500 data from Wikipedia url 
    Returns:
        df: Pandas Dataframe with the values from the url.
    """    

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

a = get_sp500_tickers()


stocks = []
sector = []
close = []
high = []
low = []
values = []

for stock in a['Symbol']:
    try:
        print(f'Fetching...{stock}')
        time.sleep(1)
        data = pdr.get_data_yahoo(stock,from_date, to_date)['Adj Close']
        last = data[-1]
        max = data.max()
        lowest = data.min()        
        diff = (last/max)
        perDiff = ((diff -1) * 100)
        values.append(perDiff)
        high.append(max)
        low.append(lowest)
        close.append(last)

    except Exception as e:
        print(f'Error on fetching {stock}')
        print(e)
        values.append(math.nan)
        high.append(math.nan)
        low.append(math.nan)
        close.append(math.nan)
        pass


a['52W High [%]'] = values
a['Highest'] = high
a['Lowest'] = low
a['Last'] = close


media = a['52W High [%]'].mean()
labels = a['Symbol']

difference = plt.scatter(a.index, a['52W High [%]'], color='r')
plt.title('S&P_500 Stocks % Difference over 52W High')
plt.axhline(y=0, color='r', linestyle='-')
plt.axhline(y=media, color='b', linestyle='--', label=(f'Mean: {media:.2f} %'))
plt.legend()
plt.xticks(a.index,  labels, rotation=90)
plt.locator_params(axis='x', nbins=len(a)/5)
plt.ylabel('52W High [%]')
plt.ylim(a['52W High [%]'].min()-5 , 5)


# Save to csv
a.to_csv('SPvalues.csv')

plt.show()
