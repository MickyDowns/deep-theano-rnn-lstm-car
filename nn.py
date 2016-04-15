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
    best_action_model = Sequential()

    # First layer.
    best_action_model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    best_action_model.add(Activation('relu'))
    best_action_model.add(Dropout(0.2))

    # Second layer.
    best_action_model.add(Dense(params[1], init='lecun_uniform'))
    best_action_model.add(Activation('relu'))
    best_action_model.add(Dropout(0.2))

    # Output layer.
    best_action_model.add(Dense(num_outputs, init='lecun_uniform'))
    best_action_model.add(Activation('linear'))

    rms = RMSprop()
    best_action_model.compile(loss='mse', optimizer=rms)

    if load:
        best_action_model.load_weights(load)

    return best_action_model


def speed_net(num_inputs, params, num_outputs, load=''):
    best_action_model = Sequential()
    
    # First layer.
    best_action_model.add(Dense(params[0], init='lecun_uniform', input_shape=(num_inputs,)))
    best_action_model.add(Activation('relu'))
    best_action_model.add(Dropout(0.3))
    
    # Second layer.
    best_action_model.add(Dense(params[1], init='lecun_uniform'))
    best_action_model.add(Activation('relu'))
    best_action_model.add(Dropout(0.3))
    
    # Third layer.
    best_action_model.add(Dense(params[2], init='lecun_uniform'))
    best_action_model.add(Activation('relu'))
    best_action_model.add(Dropout(0.3))
    
    # Output layer.
    best_action_model.add(Dense(num_outputs, init='lecun_uniform'))
    best_action_model.add(Activation('linear'))
    
    rms = RMSprop()
    best_action_model.compile(loss='mse', optimizer=rms)
    
    if load:
        best_action_model.load_weights(load)
    
    return best_action_model


def lstm_net_1(num_inputs, params, num_outputs, load=''):
    best_action_model = Sequential()
    best_action_model.add(LSTM(output_dim=params[0], input_dim=num_inputs, return_sequences=True))
    best_action_model.add(Dropout(0.2))
    best_action_model.add(LSTM(output_dim=params[1], input_dim=params[0], return_sequences=False))
    best_action_model.add(Dropout(0.2))
    best_action_model.add(Dense(output_dim=num_outputs, input_dim=512))
    best_action_model.add(Activation("linear"))
    best_action_model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return best_action_model

def lstm_net_1(num_inputs, params, num_outputs, load=''):
    best_action_model = Sequential()
    model.add(LSTM(output_dim=params[0], activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=params[1], activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model
