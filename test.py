import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from binbot import Binbot
from helper import *
import sys

stock_name = "AAPL"
model_name = "model_60"
model = load_model("./models/"+model_name)
window_size = model.layers[0].input.shape.as_list()[1]

bot = Binbot(window_size, True, model_name)
_,data = get_data(stock_name)

batch_size = 32
state = get_state(data, 0, window_size + 1 )
total_gain = 0

fig, linechart = plt.subplots()
tracker =[]
idx = 0

bot.inventory = []
l = (len(data) - 1)
for t in range(l):
    action = bot.act(state)
    next_state = get_state(data, t+1 , window_size +1)
    reward = 0
    if action == 1:
        bot.inventory.append(data[t])
        print("Buy : " + str(data[t]))
        tracker.append((idx, data[t], 'Buy'))
    
    elif action == 2 and len(bot.inventory) >0:
        strike_price = bot.inventory.pop(0)
        reward = max(data[t] - strike_price,0)
        total_gain += data[t] - strike_price
        print("Sell : " + str(data[t]) + " Current Total Gain :" + str(total_gain))
        tracker.append((idx, data[t], 'Sell'))

    idx += 1  
    done = True if t == l-1 else False
    bot.memory.append((state, action, reward, next_state, done))
    state = next_state
    if done:
        print("-"*10)
        print("stock_name " + stock_name + " total gain : "  + str(total_gain))

    if len(bot.memory) > batch_size:
        bot.expReplay(batch_size)

tracker = np.array(tracker)
linechart.plot(tracker[:,0], tracker[:,1])
plt.show()
