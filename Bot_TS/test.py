import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from binbot import Binbot
import pandas as pd
import numpy as np
from helper import get_data, get_return, get_state, get_TS_data, action_plot, benchmark

import sys

# Firstly, we use the trained network to predict actions 
stock_name = "AAPL"
# model_name = "model_" + stock_name
model_name = "model_APPL"

TS_predictor = load_model("./models/TS_AAPL")

window_size = 10

bot = Binbot(state_size=window_size, is_test=True, nn_epochs=5, model_name=model_name)
_, data = get_data(stock_name)
print("Number of days we are playing", len(data))
batch_size = 32
state = get_state(data, 0, window_size + 1)
TS = get_TS_data(data, 0, 2*window_size +1).T
state = [state, TS_predictor.predict(TS)]
print(state)
total_gain = 0 

tracker = {'close': [], 'action': [], 'reward': []}
idx = 0

bot.buffer = []
l = (len(data) - 1)
print(l)
for t in range(l):
    action = bot.act(state)

    next_state = get_state(data, t+1, window_size + 1)
    next_TS = get_TS_data(data, t+1, 2*window_size +1).T
    next_state = [next_state, TS_predictor.predict(next_TS)]

    reward = 0
    tracker['close'].append(data[t])
    if action == 1:
        bot.buffer.append(data[t])
        print("Buy at {:.3f}$".format(data[t]))
        tracker['action'].append("Buy")

    elif action == 2 and len(bot.buffer) > 0:

        hold = True
        for i in range(len(bot.buffer)):
            buy_price = bot.buffer[i]
            if (data[t] - buy_price) > 0:
                buy_price = bot.buffer.pop(i)
                print("Sell at {:.3f}$, Single bet gain:{:.3f}$, Current Total Gain:{:.3f}$".format(data[t], 
                                                                    data[t] - buy_price, 
                                                                    total_gain))
                hold = False
                total_gain += data[t] - buy_price
                reward = max(data[t] - buy_price, 0)
                tracker['action'].append("Sell")
                break

        if hold:
            print("Prefer to hold")
            tracker['action'].append("Hold")

    else:
        tracker['action'].append("Hold")
    tracker['reward'].append(reward)
    is_complete = True if t == l-1 else False
    bot.memory.append((state, action, reward, next_state, is_complete))

    state = next_state

    if is_complete:
        print("-"*10)
        print("stock_name {}, total gain:{:.3f}".format(stock_name, total_gain) )

    if len(bot.memory) > batch_size:
        bot.replay(batch_size)


# Second: We evaluate our bot.
action_plot(tracker,l)

tracker = pd.DataFrame(tracker)

tracker['rate_return'] = tracker['reward']/(tracker['close']-tracker['reward'])
series = list(tracker[tracker['action'] == "Sell"]['rate_return'])
rate_return_avg = np.mean(series)
sharpe_ratio =(rate_return_avg-0.0155)/np.sqrt(np.var(series))
print("The avg rate return is {:.2f}%, the sharpe ratio is {:.3f}".format(100*rate_return_avg, sharpe_ratio))

benchmark()