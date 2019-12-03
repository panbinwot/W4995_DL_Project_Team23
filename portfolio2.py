import matplotlib.pyplot as plt
from binbot import Binbot
import pandas as pd
import numpy as np
from helper import get_data, get_return, get_state, action_plot, benchmark_sp, get_stock_names
import sys
from keras.models import load_model
from helper import generate_buffer, get_test_dct, evaluate
import seaborn as sns

stock_lst =[x.split(" ")[0] for x in get_stock_names()]
test_dct = get_test_dct(stock_lst)

# Initialize shares uniformly
shares = [10000/(test_dct[stock][0]*(1+len(stock_lst))) for stock in stock_lst ]
print("Init Shares")
print(shares)

stock_lst =[x.split(" ")[0] for x in get_stock_names()]
acts = generate_buffer(stock_lst)
idx = 0
res = []
print("Test begins:")
_, benchmark = get_data('SP500', verbose = 1)
duration = len(benchmark)-1

# for stock in stock_lst:
#     print("-"*30)
#     print("Running on Stock"+stock)
#     stock_name = stock
#     # model_name = "model_" + stock_name
#     window_size = 10
#     bot = Binbot(window_size, True, model_name= "model_APPL_short_term")
#     _, data = get_data(stock_name)
#     print("Number of days we are playing", len(data))
#     batch_size = 32
#     state = get_state(data, 0, window_size + 1)
#     print(state)
#     total_gain = 0 

#     tracker = {'close': [], 'action': [], 'reward': []}
#     idx = 0

#     bot.buffer = []
#     l = (len(data) - 1)
#     for t in range(l):
#         action = bot.act(state)
#         next_state = get_state(data, t+1, window_size + 1)
#         reward = 0
#         tracker['close'].append(data[t])
#         if action == 1:
#             bot.buffer.append(data[t])
#             print("Buy at {:.3f}$".format(data[t]))
#             tracker['action'].append("Buy")

#         elif action == 2 and len(bot.buffer) > 0:
#             buy_price = bot.buffer.pop(0)
#             reward = max(data[t] - buy_price, 0)
#             total_gain += data[t] - buy_price
#             print("Sell at {:.3f}$, Single bet gain:{:.3f}$, Current Total Gain:{:.3f}$".format(data[t], 
#                                                                     data[t] - buy_price, 
#                                                                     total_gain))
#             tracker['action'].append("Sell")
#         else:
#             tracker['action'].append("Hold")
#         tracker['reward'].append(reward)
#         is_complete = True if t == l-1 else False
#         bot.memory.append((state, action, reward, next_state, is_complete))
#         state = next_state
#         if is_complete:
#             print("-"*10)
#             print("stock_name {}, total gain:{:.3f}".format(stock_name, total_gain) )

#         if len(bot.memory) > batch_size:
#             bot.replay(batch_size)
#         acts[stock].append(action)

# np.save('./models/actions_2.npy', acts) 

acts = np.load('./models/actions_2.npy',allow_pickle='TRUE').item()

buffer = generate_buffer(stock_lst)
cash = (10000/(1+len(stock_lst)))

for d in range(duration):
    # if d>4: break
    actions = []
    total_value = 0
    reward_d = 0
    total_value = 0
    for i, stock in enumerate(stock_lst,0):
        # print("_"*30)
        # print("Run on day {}, iteration: {}/{}, running on stock {}".format(d,idx,duration*29,stock))
        data = test_dct[stock]
        action = acts[stock][d]
        if action == 1:
            buffer[stock].append(data[d])
            # print("Buy at {:.3f}$".format(data[d]))
        if action == 2 and len(buffer[stock]) >0:
            # For now, we are not alow the computer to sell short. 
            buy_price = buffer[stock].pop(0)
            # print("Sell at {:.3f}$, Single bet gain:{:.3f}$".format(data[d],reward2))
            cash += data[d]*shares[i]     
        actions.append(action) 
        idx += 1

    # Update Shares
    if cash>0 and actions.count(1)>=1:
        for i, stock in enumerate(stock_lst,0):
            if actions[i] == 1:
                shares[i] += (0.5*cash)/(test_dct[stock][d]*actions.count(1))
            if actions[i] == 2:
                shares[i] = 0
        cash = 0.5*cash
    
    today_lst = [test_dct[stock][d] for stock in stock_lst ] 
    total_value = np.array(shares).dot(np.array(today_lst))
    print("Test day {}, cash is {:.2f}$, portfolio value is {:.2f}$".format(d,
                                                                cash,
                                                                total_value+cash))
    # print(shares)
    print(actions)
    res.append(total_value+cash)


benchmark = np.array(benchmark)[101:]
rate_avg, sharpe = evaluate(res)
res = np.array(res)[100:]

print("rate of return avg {}, sharp ratio {}".format(rate_avg, sharpe))
x = [i+1 for i in range(len(res))]
sns.lineplot(x, benchmark/benchmark[0], label = "SP500")
sns.lineplot(x, res/res[0], label = "Agent")
plt.xlabel("Days Testing")
plt.ylabel("Value (standardized)")
plt.legend()
plt.show()