import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from binbot import Binbot
import pandas as pd
from helper import *
import sys

# Firstly, we use the trained network to predict actions 
stock_name = "AAPL"
# model_name = "model_" + stock_name
model_name = "model_60"
model = load_model("./models/"+model_name)
window_size = model.layers[0].input.shape.as_list()[1]

bot = Binbot(window_size, True, model_name)
_, data = get_data(stock_name)
print("Number of days we are playing", len(data))
batch_size = 32
state = get_state(data, 0, window_size + 1)
total_gain = 0

tracker = {'close': [], 'action': [], 'reward': []}
idx = 0

bot.inventory = []
l = (len(data) - 1)
for t in range(l):
    action = bot.act(state)
    next_state = get_state(data, t+1, window_size + 1)
    reward = 0
    tracker['close'].append(data[t])
    if action == 1:
        bot.inventory.append(data[t])
        print("Buy : " + str(data[t]))
        tracker['action'].append("Buy")

    elif action == 2 and len(bot.inventory) > 0:
        strike_price = bot.inventory.pop(0)
        reward = max(data[t] - strike_price, 0)
        total_gain += data[t] - strike_price
        print("Sell : " + str(data[t]) + 
                 "Single bet gain:" + str(reward)+
              " Current Total Gain :" + str(total_gain))
        tracker['action'].append("Sell")
    else:
        tracker['action'].append("Hold")
    tracker['reward'].append(reward)
    is_complete = True if t == l-1 else False
    bot.memory.append((state, action, reward, next_state, is_complete))
    state = next_state
    if is_complete:
        print("-"*10)
        print("stock_name " + stock_name + " total gain : " + str(total_gain))

    if len(bot.memory) > batch_size:
        bot.replay(batch_size)


# Second: We evaluate our bot.
action_plot(tracker,l)

tracker = pd.DataFrame(tracker)

tracker['rate_return'] = tracker['reward']/(tracker['close']-tracker['reward'])
series = list(tracker[tracker['action'] == "Sell"]['rate_return'])
rate_return_avg = len(series)*np.mean(series)
sharpe_ratio =(rate_return_avg-0.0155)/np.sqrt(np.var(series))
print("The avg rate return is {}, the sharpe ratio is, {}".format(rate_return_avg, sharpe_ratio))

benchmark()


