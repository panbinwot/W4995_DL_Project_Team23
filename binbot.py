# Bin is trying to replicate the deep q-learning framework he saw online
# he saw online. 11/01/2019
# The trading agent is using Deep Q-learning Network. 
# * The neural network is implement in keras.
# DQNagent: https://keon.io/deep-q-learning/
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Binbot:
    def __init__(self, state_size, is_test=False, model_name = ""):
        self.state_size = state_size
        self.action_size = 3 # Define Actions for the bot: Hold, Buy and Sell
        self.memory = deque(maxlen = 1000)
        self.inventory = []
        self.model_name = model_name
        self.is_test = is_test
        self.first_visit = True
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model("models/" + model_name) if is_test else self._model()
    
    def _model(self):
    '''
    This is the Nerual Network Part. We use a NN to approximate the value of the value function.
    '''
        model = Sequential()
        model.add(Dense(units = 64, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(units = 32, activation="relu"))
        model.add(Dense(units = 16, activation="tahn"))
        model.add(Dense(units = 8, activation="relu"))
        model.add(Dense(self.action_size, activation = "linear") )
        model.compile(loss = "mse", optimizer = Adam(lr = 0.001))
        return model

    def act(self, state):
        if self.is_test and self.first_visit:
            self.first_visit = False
            return 1
        if not self.is_test and np.random.rand()<= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    
    def dp(self, batch_size):
        '''
        This is the dynamic programming part.
        '''
        mini_batch = []
        l = len(self.memory)

        for i in range(l - batch_size +1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            target =  reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Here we train the 
            self.model.fit(state, target_f, epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


