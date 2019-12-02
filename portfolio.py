import matplotlib.pyplot as plt
from binbot2 import Binbot2
import pandas as pd
import numpy as np
from helper import get_data, get_return, get_state, action_plot, benchmark, get_stock_names
import sys
from keras.models import load_model
from helper import generate_buffer, get_test_dct
import seaborn as sns

'''
Shares is how we decide to distribute the money
We initialize it by uniformly distributed over stocks (including cash)
We also initialize the net worth of the portfolio as 1$ (Naive Setting)
'''
stock_lst =[x.split(" ")[0] for x in get_stock_names()]
test_dct = get_test_dct(stock_lst)

# Initialize shares uniformly
shares = [10000/(test_dct[stock][0]*(1+len(stock_lst))) for stock in stock_lst ]
print("Init Shares")
print(shares)

print("Test begins:")
_, data = get_data('AAPL', start_test = '2015-12-31 12:00:00', verbose = 1)
duration = len(data)-1
window_size = 10
bot = Binbot2(state_size = window_size,stock_names = stock_lst ,is_test=True)

idx = 0
res = []

current_state = generate_buffer(stock_lst)
next_state = generate_buffer(stock_lst)
# Firstly, we initialize state zero
for stock in stock_lst:
    # current_state[stock] = get_state(test_dct[stock], 0, window_size + 1)
    current_state[stock] =np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

cash = (10000/(1+len(stock_lst)))
for d in range(duration):
    if d>3: break
    actions = []
    total_value = 0
    for i, stock in enumerate(stock_lst,0):
        bot.current = stock
        print("_"*30)
        print("Run on day {}, iteration: {}/{}, running on stock {}".format(d,idx,duration*29,stock))
        data = test_dct[stock]
        # try:
        # bot.model_name = 'model'+str(stock)
        bot.model_name = 'model_80'
        next_state[stock] = get_state(data, t = d+1, n = window_size+1 )
        action = bot.act(current_state[stock])
        reward = 0.0
        if action == 1:
            bot.buffer[stock].append(data[d])
            print("Buy at {:.3f}$".format(data[d]))
        if action == 2 and len(bot.buffer[stock]) >0:
            buy_price = bot.buffer[stock].pop(0)
            reward = max(data[d] - buy_price, 0)
            reward2 = (data[d] - buy_price)*shares[i]
            
            print("Sell at {:.3f}$, Single bet gain:{:.3f}$".format(data[d],reward2))
            cash += data[d]*shares[i]

        # is_complete = True if d == duration-1 else False
        # bot.memory[stock].append((current_state[stock], action, reward, next_state[stock], is_complete))
        # current_state[stock] = next_state[stock]
        # if len(bot.memory[stock]) > 32:
            # bot.replay()       

        actions.append(action)
        idx += 1

    print(actions)
    # Update Shares
    if cash>0 and actions.count(1)>=1:
        for i, stock in enumerate(stock_lst,0):
            if actions[i] == 1:
                shares[i] += (0.5*cash)/(test_dct[stock][d]*actions.count(1))
            if actions[i] == 2:
                shares[i] = 0.0
        cash = 0.5*cash
    
    today_lst = [test_dct[stock][d] for stock in stock_lst ] 
    total_value = np.array(shares).dot(np.array(today_lst))
    print("Test day {}, cash is {:.2f}$, portfolio value is {:.2f}$".format(d,
                                                                cash,
                                                                total_value+cash))
    print(shares)
    res.append(total_value+cash)

x = [i+1 for i in range(len(res))]
sns.lineplot(x, res)


    
    


