"""
Once a model is learned, use this to play it.
"""

from flat_game import carmunk
import numpy as np
from nn import neural_net
import random

NUM_INPUT = 8
NUM_OUTPUT = 6
START_SPEED = 50
START_ACTION = 0
START_DISTANCE = 0


def play(model):

    car_distance = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    state, _, speed, _, _, _ = game_state.frame_step(START_ACTION, START_SPEED, START_DISTANCE)

    # Move.
    while True:
        car_distance += 1
        
        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
        
        # Take action.
        state, _, speed, _, _, _ = game_state.frame_step(action, speed, car_distance)
        
        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    saved_model = 'saved-best_action_models/240-160-100-50000-100000.h5'
    model = neural_net(NUM_INPUT, [240, 160, 80], NUM_OUTPUT, saved_model)
    play(model)
