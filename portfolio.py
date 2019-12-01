import matplotlib.pyplot as plt
from binbot2 import Binbot2
import pandas as pd
import numpy as np
from helper import get_data, get_return, get_state, action_plot, benchmark, get_stock_names
import sys
from keras.models import load_model
from helper import generate_buffer, get_test_dct

'''
Weights is how we decide to distribute the money
We initialize it by uniformly distributed over stocks (including cash)
We also initialize the net worth of the portfolio as 1$ (Naive Setting)
'''
stock_lst =[x.split(" ")[0] for x in get_stock_names()]

weights = [1/(1+len(stock_lst))]*(1+len(stock_lst))
print(len(weights))
print("Test begins:")
_, data = get_data('AAPL', start_test = '2015-12-31 12:00:00', verbose = 1)
duration = len(data)-1

window_size = 10
bot = Binbot2(state_size = window_size,stock_names = stock_lst ,is_test=True)

idx = 0

current_state = generate_buffer(stock_lst)
next_state = generate_buffer(stock_lst)

test_dct = get_test_dct(stock_lst)

# Firstly, we initialize state one
for stock in stock_lst:
    # current_state[stock] = get_state(test_dct[stock], 0, window_size + 1)
    current_state[stock] =np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
total_gain = 0

for d in range(duration):
    if d>3: break
    delta_weight = np.array([0]*(1+len(stock_lst)))
    rewards,actions = [],[]
    delta = 0
    for i, stock in enumerate(stock_lst,0):
        bot.current = stock
        print("_"*30)
        print("Run on day {}, iteration: {}/{}, running on stock {}".format(d,idx,duration*30,stock))
        if stock == "DOW": 
            actions.append(0)
            rewards.append(0)
            continue
        data = test_dct[stock]
        # try:
        # bot.model_name = 'model'+str(stock)
        bot.model_name = 'model_70'
        next_state[stock] = get_state(data, t = d+1, n = window_size+1 )
        action = bot.act(current_state[stock])
        reward = 0
        if action == 1:
            bot.buffer[stock].append(data[d])
            print("Buy at {:.3f}$".format(data[d]))
        elif action ==2 and len(bot.buffer[stock]) >0:
            buy_price = bot.buffer[stock].pop(0)
            reward = max(data[d] - buy_price, 0)
            reward2 = data[d] - buy_price
            total_gain += data[d] - buy_price
            print("Sell at {:.3f}$, Single bet gain:{:.3f}$, Current Total Gain:{:.3f}$".format(data[d], 
                                                                reward2, 
                                                                total_gain))

            delta += weights[i]
            weights[i] = 0
        is_complete = True if d == duration-1 else False

        bot.memory[stock].append((current_state[stock], action, reward, next_state[stock], is_complete))
        current_state[stock] = next_state[stock]
        if len(bot.memory[stock]) > 32:
            bot.replay()                
        actions.append(action)
        rewards.append(reward)
        # except:
        # print("Bot action failed")
        # actions.append(3)
        # rewards.append(0)
    # net_worth += np.array(rewards)
        idx += 1
    print(delta)
    print(actions)
    if actions.count(1)>1:
        weights = [weights[x] + delta/actions.count(1)  if actions[x]==1 else weights[x] for x in range(len(stock_lst))]
    print("Test day {}, the net worth of the portfolio is {:.2f}$".format(d,100))
    print(weights)

    
    


