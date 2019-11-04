import numpy as np
import math

def sigmoid(x):
    return 1 / (1+math.exp(-x))

def get_state(data, t, n):
    d = t - n + 1
    block = data[d:t+1] if d >= 0 else -d*[data[0]] + data[0:t+1] 
    res = []
    for i in range(n-1):
        res.append(sigmoid(block[i+1] - block[i]))
    return np.array([res])