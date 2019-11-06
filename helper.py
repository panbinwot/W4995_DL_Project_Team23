import numpy as np
import pandas as pd
import math

def get_data(stock_name):
    df = pd.read_csv("./data/"+stock_name+".csv")
    train = list(df[(df['timestamp'] >= '2010-1-1 01:00:00') & (df['timestamp'] <= '2015-12-31 04:00:00')]['close'])
    test = list(df[(df['timestamp'] >= '2016-1-1 01:00:00') & (df['timestamp'] <= '2018-12-31 12:00:00')]['close'])
    return train, test

def get_state(data, t, window_size):
    # This function takes in inputs and return the states for MDP
    d = t - window_size
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    return get_return(block,  window_size-1, True)

def evaluate(series):
    '''
    Input: A list of returns
    The function evalutes a series of close price by computing the average rate of return and sharpe ratio.
    Output: Rate of Return, Sharpe Ratio
    * The risk free rate is set as 1.55
    '''
    series = np.array(series)
    rate_returns = get_return(series, len(series), True)
    rate_return_avg = np.mean(rate_returns)
    sharpe_ratio =(rate_return_avg-1.55)/np.var(rate_returns)
    return rate_return_avg, sharpe_ratio

def get_return(lst,n, m ):
    res = []
    for i in range(n-1):
        # Computing the return rate
        # An alternative way is 
        if m :
            res.append(sigmoid(lst[i - 1] - lst[i]))
        else :
            res.append((lst[i - 1] - lst[i])/lst[i-1])
            
    return np.array(res)

def sigmoid(x):
    return 1 / (1+math.exp(-x))
