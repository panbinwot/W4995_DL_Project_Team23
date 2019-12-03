# This bot plays on a portfolio instead of single stock.
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import keras.optimizers as optimizers

import numpy as np
import random
from collections import deque

class Binbot2:
    def __init__(self,state_size, stock_names, nn_epochs = 5,is_test=False):
        self.state_size = state_size
        self.action_size = 3 # Define Actions for the bot: Hold, Buy and Sell
        self.memory = self.generate_memory(stock_names)
        self.buffer = self.generate_buffer(stock_names)
        self.model_name = 'model_10'
        self.is_test = is_test
        self.first_visit = True
        self.gamma = 0.95
        self.epsilon = 1.0
        self.learning_rate = 0.001
        self.nn_epochs = nn_epochs
        self.current = ""
    def _model(self):
        # DQN use a neural network to approximate the Q Values
        model = Sequential()
        model.add(Dense(units = 64, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(units = 32, activation="relu"))
        model.add(Dense(units = 8, activation="relu"))
        model.add(Dense(self.action_size, activation = "linear") )
        model.compile(loss="mse", optimizer=optimizers.SGD(lr = 0.01))
        return model

    def act(self, state):
        self.model = load_model("models/" + self.model_name) if self.is_test else self._model()
        #  If we are testing, a Buy action will be initialized.
        if self.is_test and self.first_visit:
            self.first_visit = False
            return 1
        
        # Allow the bot chooses random action with prob epsilon. 
        # Epsilon starts with 1 but we set it decreases by 50% each time. 

        if not self.is_test and np.random.rand()<= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
     
        return np.argmax(options[0])
    
    def replay(self, batch_size = 32):
        '''
        This is the dynamic programming part.
        The replay is the most siginficant part of DQN. 
        We are adding historical information into the learning process.
        '''
        mini_batch = random.sample(self.memory[self.current], batch_size)
        mini_batch, l = [], len(self.memory[self.current])
        for i in range(l -batch_size +1, l):
            mini_batch.append(self.memory[self.current][i])
        
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Here we train the NN
            self.model.fit(state, target_f, epochs = self.nn_epochs, verbose = 0)

        if self.epsilon > 0.01:
            self.epsilon *= 0.995


#-----------------some helper methods of the class---------------
    def generate_buffer(self, stock_names):
        '''
        Use to create a buffer to track the buying price for each stock,].
        '''
        dct = {}
        for stock in stock_names:
            dct[stock] = []
        return dct

    def generate_memory(self, stock_names):
        '''
        Use to create a buffer to track the buying price for each stock,].
        '''
        dct = {}
        for stock in stock_names:
            dct[stock] = deque(maxlen = 1000)
        return dct