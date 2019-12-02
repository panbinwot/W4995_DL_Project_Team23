import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from binbot import Binbot
import pandas as pd
import numpy as np
from helper import get_data, get_return, get_state, action_plot, benchmark
import seaborn as sns
import sys


res2 = []

# Firstly, we use the trained network to predict actions 
stock_name = "KO"
# model_name = "model_" + stock_name
window_size = 10
_, benchmark = get_data('SP500', verbose = 0)
bot = Binbot(state_size = window_size, is_test = True, model_name= "model_80")
_, data = get_data(stock_name)

share = 5000/data[0]
cash = 5000
print("Number of days we are playing", len(data))
batch_size = 32
state = get_state(data, 0, window_size + 1)
print(state)
total_gain = 0 

tracker = {'close': [], 'action': [], 'reward': []}
idx = 0

bot.buffer = []
l = (len(data) - 1)
for t in range(l):
    print("Share.{}".format(share))
    action = bot.act(state)
    next_state = get_state(data, t+1, window_size + 1)
    reward = 0
    tracker['close'].append(data[t])
    if action == 1:
        bot.buffer.append(data[t])
        print("Buy at {:.3f}$".format(data[t]))
        tracker['action'].append("Buy")
        if cash>0:
            share += 0.5*cash/data[t]
            cash = cash - 0.5*cash

    elif action == 2 and len(bot.buffer) > 0:
        buy_price = bot.buffer.pop(0)
        reward = max(data[t] - buy_price, 0)
        total_gain += data[t] - buy_price
        print("Sell at {:.3f}$, Single bet gain:{:.3f}$, Current Total Gain:{:.3f}$".format(data[t], 
                                                                data[t] - buy_price, 
                                                                total_gain))
        tracker['action'].append("Sell")
        
        cash += data[t]*share
        share = 0
    else:
        print("We hold for today")
        tracker['action'].append("Hold")
    tracker['reward'].append(reward)
    # is_complete = True if t == l-1 else False
    # bot.memory.append((state, action, reward, next_state, is_complete))
    state = next_state
    # if is_complete:
    #     print("-"*10)
    #     print("stock_name {}, total gain:{:.3f}".format(stock_name, total_gain) )

    # if len(bot.memory) > batch_size:
    #     bot.replay(batch_size)
    
    total_value = cash + share*data[t]
    res2.append(total_value)
# Second: We evaluate our bot.
# action_plot(tracker,l)

tracker = pd.DataFrame(tracker)

tracker['rate_return'] = tracker['reward']/(tracker['close']-tracker['reward'])
series = list(tracker[tracker['action'] == "Sell"]['rate_return'])
rate_return_avg = np.mean(series)
sharpe_ratio =(rate_return_avg-0.0155)/np.sqrt(np.var(series)+0.001)
print("The avg rate return is {:.2f}%, the sharpe ratio is {:.3f}".format(100*rate_return_avg, sharpe_ratio))

# benchmark()

x = [i+1 for i in range(len(res2))]
res2 = np.array(res2)
benchmark = np.array(benchmark)[1:]
plt.plot(x, benchmark/benchmark[0], label = "SP500")
plt.plot(x, res2/res2[0], label = "Agent")
plt.legend()
plt.show()
