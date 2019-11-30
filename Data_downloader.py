import pandas as pd
import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime


def dataloader(stock_name,
              start = '1995-01-01 01:00:00',
              end = '2018-12-31 04:00:00',verbose = True):

  my_share = share.Share(stock_name)
  symbol_data = None

  try:
      symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,30,
                                            share.FREQUENCY_TYPE_DAY,1)
      if verbose:
        print(stock_name+" has been downloaded")
      
  except YahooFinanceError as e:
      print(e.message)
      sys.exit(1)
  df = pd.DataFrame(symbol_data)
  
  df['timestamp'] = [datetime.fromtimestamp(t/1000) for t in df['timestamp']]

  # test = df[(df['timestamp'] >= '2016-1-1 01:00:00') & (df['timestamp'] <= '2018-12-31 04:00:00')]
  return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

# djia_str = "KO MCD JPM AXP CAT UTX MSFT CVX AAPL HD DIS V XOM TRV PFE MRK JNJ BA VZ IBM GS WBA INTC DOW UNH WMT PG CSCO NKE MMM"
# name_lst = [x for x in djia_str.split(" ")]

# for comp in name_lst:
#   df = dataloader(comp)
#   df.to_csv("./data/"+comp+".csv")
# dataloader('^GSPC').to_csv("./data/"+"SP500"+".csv")
