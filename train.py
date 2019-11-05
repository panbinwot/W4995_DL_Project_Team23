from binbot import Binbot
from helper import *
import pandas as pd
import numpy as np

window_size = 10
data,_ = get_data("AAPL",window_size)
stock_name, episode_count = 'APPL' , int(len(data)/window_size)

bot = Binbot(window_size)

l = len(data) 
batch_size = 32

for e in range(episode_count + 1):
    print("-"*20)
    print("Episode " + str(e)+"/"+ str(episode_count))
    state = get_state(data, 0, window_size +1)

    total_gain = 0
    bot.inventory = []
    for t in range(l):
        action = bot.act(state)

        next_state = get_state(data, t+1, window_size +1)
        reward = 0

        if action == 1: 
            bot.inventory.append(data[t])
            print("Buy at"+ str(data[t]))
        elif action==2 and len(bot.inventory)>0:
            bought_price = bot.inventory.pop(0)
            reward = max(data[t]-bought_price,0)
            total_gain += data[t] - bought_price
            print("Sell at "+str(data[t])+" Current gain:"+str(total_gain))

        done = True if t == l - 1 else False
        bot.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("All done, the total gain for this round is "+ str(total_gain))
        
        if len(bot.memory) > batch_size:
            bot.expReplay(batch_size)
    if e % 10 == 0:
        bot.model.save("./models/model_"+str(e))


