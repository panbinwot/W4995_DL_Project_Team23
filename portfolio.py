import matplotlib.pyplot as plt
from binbot import Binbot
import pandas as pd
import numpy as np
from helper import get_data, get_return, get_state, action_plot, benchmark, get_stock_names
import sys
from keras.models import load_model

stock_lst = get_stock_names()

'''
Weights is how we decide to distribute the money
We initialize it by uniformly distributed over stocks (including cash)
We also initialize the net worth of the portfolio as 1$ (Naive Setting)
'''
data_test = pd.read_csv('./data/test.csv')
stock_lst = list(data_test.keys() )
net_worth = 10000
shares = list((net_worth/len(stock_lst))/data_test.loc[0,])

weights = [1/(1+len(stock_lst))]*(1+len(stock_lst))
print(len(weights))
print("Test begins:")
for d in range(152):
    if d>2: break
    delta_weight = np.array([0]*(1+len(stock_lst)))
    rewards,actions = [],[]
    delta = 0
    for i, stock in enumerate(stock_lst,0):
        # if i>2: break
        # model = load_model("./models/"+stock)
        print("_"*30)
        print("running on stock "+ stock)
        data = list(data_test[stock])
        model_name = "model_10"
        model = load_model('./models/'+model_name)
        window_size = model.layers[0].input.shape.as_list()[1]
        bot = Binbot(window_size, is_test=True, model_name = model_name)
        batch_size = 32
        state = get_state(data, t = d, n = window_size+1 )

        action = bot.act(state)
        print("the action is {}".format(action))
        reward = 0
        if action == 1:
            bot.buffer.append(data[d])
        elif action ==2 and len(bot.buffer) >0:
            buy_price = bot.buffer.pop(0)
            reward = data[d] - buy_price
            delta += weights[i]
            weights[i] = 0
        actions.append(action)
        rewards.append(reward)
    # net_worth += np.array(rewards)
    weights = [weights[x] + delta/actions.count(1)  if actions[x]==1 else weights[x] for x in range(len(stock_lst))]
    print("Test day {}, the net worth of the portfolio is {:.2f}$".format(d,net_worth))
    print(weights)

    
    


