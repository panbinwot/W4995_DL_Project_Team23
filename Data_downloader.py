import pandas as pd
import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime
import matplotlib.pyplot as plt

def dataloader(symbol):
  my_share = share.Share(symbol)
  symbol_data = None

  try:
      symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,30,
                                            share.FREQUENCY_TYPE_DAY,1)
      print(symbol+" has been downloaded")
      
  except YahooFinanceError as e:
      print(e.message)
      sys.exit(1)
  df = pd.DataFrame(symbol_data)
  
  df['timestamp'] = [datetime.fromtimestamp(t/1000) for t in df['timestamp']]

  return df[(df['timestamp'] >= '1995-1-1 01:00:00') & (df['timestamp'] <= '2018-12-31 04:00:00')]
  

djia_str = "KO MCD JPM AXP CAT UTX MSFT CVX AAPL HD DIS V XOM TRV PFE MRK JNJ BA VZ IBM GS WBA INTC DOW UNH WMT PG CSCO NKE MMM"
name_lst = [x for x in str.split(" ")]

apple = dataloader("AAPL")
nike =  dataloader("NKE")
dji  = dataloader("DJI")

# print(apple['close'].describe() )
# print(nike['close'].describe() )
# print(dji['close'].describe() )
apple.head()
