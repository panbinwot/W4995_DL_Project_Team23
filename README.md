# Term Project for W4995 Sec:10 Deep Learning. 
The purpose of the project is building a smart trading agent with deep reinforcement learning. We form the financial time series (stock price data) as Markov Descision Proess (MDP). Although the price of $T_{n}$ is not a sufficient statistic of $$T_{n+1}$$, we can include history data to the process close to MDP.

The agent we build will compete with S&P 500 Index and Dow Jones Index in terms of rate of return and Sharpe Ratio, which evalute a trading strategy's return and risk.

At this point, the project is not finished and develop two separate ideas.

The first one is a trading agent you can see on binbot.py and train through train.py\\
The second idea is the deterministic prediction of time series you can run through Project_TS.ipynb
