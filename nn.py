"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def turn_net(num_inputs, params, num_outputs, load=''):
    model = Sequential()

    # First layer.
    model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(num_outputs, init='lecun_uniform'))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)

    return model


def avoid_net(num_inputs, params, num_outputs, load=''):
    model = Sequential()
    
    # First layer.
    model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Third layer.
    model.add(Dense(params[2], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Output layer.
    model.add(Dense(num_outputs, init='lecun_uniform'))
    model.add(Activation('linear'))
    
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    
    if load:
        model.load_weights(load)
    
    return model


def acquire_net(num_inputs, params, num_outputs, load=''):
    model = Sequential()
    
    # First layer.
    model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Second layer.
    #model.add(Dense(params[1], init='lecun_uniform'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(num_outputs, init='lecun_uniform'))
    model.add(Activation('linear'))
    
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    
    if load:
        model.load_weights(load)
    
    return model


def hunt_net(num_inputs, params, num_outputs, load=''):
    model = Sequential()
    
    # First layer.
    model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Third layer.
    model.add(Dense(params[2], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Output layer.
    model.add(Dense(num_outputs, init='lecun_uniform'))
    model.add(Activation('linear'))
    
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    
    if load:
        model.load_weights(load)
    
    return model


def pack_net(num_inputs, params, num_outputs, load=''):
    model = Sequential()
    
    # First layer.
    model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Second layer.
    model.add(Dense(params[1], init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Output layer.
    model.add(Dense(num_outputs, init='lecun_uniform'))
    model.add(Activation('linear'))
    
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    
    if load:
        model.load_weights(load)
    
    return model


def tbd_net(num_inputs, params, num_outputs, load=''):
    model = Sequential()
    model.add(LSTM(output_dim=params[0], input_dim=num_inputs, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=params[1], input_dim=params[0], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=params[2], input_dim=params[1], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=num_outputs, input_dim=params[2]))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model

def lstm_net_1(num_inputs, params, num_outputs, load=''):
    model = Sequential()
    model.add(LSTM(output_dim=params[0], activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=params[1], activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model
