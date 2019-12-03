from binbot import Binbot
from helper import get_data, get_return, get_state, get_TS_data, action_plot, benchmark
import pandas as pd
import numpy as np
from keras.models import load_model

window_size = 10
data,_ = get_data("AAPL")
stock_name, episodes = 'APPL' , int(len(data)/window_size)

TS_predictor = load_model("./models/TS_AAPL")

bot = Binbot(window_size)
max_tot = 0

l = len(data) 
batch_size = 32

for e in range(episodes + 1):
    print("-"*20)
    print("Episode " + str(e)+"/"+ str(episodes))
    state = get_state(data, 0, window_size +1)
    TS = get_TS_data(data, 0, 2*window_size +1).T
    state = [state, TS_predictor.predict(TS)]

    total_gain = 0
    bot.buffer = []

    for t in range(l):
        action = bot.act(state)

        next_state = get_state(data, t+1, window_size +1)
        next_TS = get_TS_data(data, t+1, 2*window_size +1).T
        next_state = [next_state, TS_predictor.predict(next_TS)]
        reward = 0

        if action == 1: 
            bot.buffer.append(data[t])
            #print("Buy at {:.3f}".format(data[t]))
        elif action==2 and len(bot.buffer)>0:

            for i in range(len(bot.buffer)):
                buy_price = bot.buffer[i]
                if (data[t] - buy_price) > 0:
                    buy_price = bot.buffer.pop(i)
                    """print("Sell at {:.3f}$, Single bet gain:{:.3f}$, Current Total Gain:{:.3f}$".format(data[t], 
                                                                        data[t] - buy_price, 
                                                                        total_gain))"""
                    total_gain += data[t] - buy_price
                    reward = max(data[t] - buy_price, 0)
                    break

        is_complete = True if t == l - 1 else False
        bot.memory.append((state, action, reward, next_state, is_complete))
        state = next_state

        if is_complete:
            print("-"*10)
            print("stock_name {}, total gain:{:.3f}".format(stock_name, total_gain) )
        
        #if len(bot.memory) > batch_size:
    
    if total_gain >= max_tot:
        bot.model.save("./models/model_"+ stock_name)
        max_tot = total_gain

    bot.replay(batch_size)


