from flat_game import carmunk
import numpy as np
import random
import csv
from nn import neural_net, LossHistory
import os.path
import timeit
import time

#NUM_FRAMES = 1 # was 2
#NUM_SENSORS = 7 # was 3
NUM_INPUT = 8 # seven distance readings + 1 speed reading
NUM_OUTPUT = 6
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.
START_SPEED = 50
START_ACTION = 0
START_DISTANCE = 0


def train_net(best_action_model, params):

    filename = params_to_filename(params)

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 500000  # Number of frames to play. was 1000000
    batchSize = params['batchSize']
    buffer = params['buffer']

    # Just stuff used below.
    max_car_distance = 0
    car_distance = 0
    t = 0
    cum_rwd = 0
    cum_rwd_read = 0
    cum_rwd_dist = 0
    cum_rwd_speed = 0
    
    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').
    save_init = True
    loss_log = []

    # Create a new game instance.
    game_state = carmunk.GameState()

    # Get initial state by doing nothing and getting the state.
    state, new_reward, cur_speed, _, _, _ = game_state.frame_step(START_ACTION, START_SPEED, START_DISTANCE)

    # frame_step returns reward, state, speed
    #state = state_frames(state, np.array([[0, 0, 0, 0, 0, 0, 0]])) # zeroing distance readings
    #state = state_frames(state, np.zeros((1,NUM_SENSORS))) # zeroing distance readings

    # Let's time it.
    start_time = timeit.default_timer()

    # Run the frames.
    while t < train_frames:

        #time.sleep(0.5)
        
        t += 1
        car_distance += 1

        # Choose an action.
        if random.random() < epsilon or t < observe:
            action = np.random.randint(0, NUM_OUTPUT)  # random
        else:
            # Get Q values for each action
            qval = best_action_model.predict(state, batch_size=1)
            # best_action_model was passed to this function. call it w/ current state
            action = (np.argmax(qval))  # best prediction

        # Take action, observe new state and get our treat.
        new_state, new_reward, new_speed, new_rwd_read, new_rwd_dist, new_rwd_speed = \
            game_state.frame_step(action, cur_speed, car_distance)
    
        # Use multiple frames.
        #new_state = state_frames(new_state, state) # seems this is appending 2-3 moves, results
        
        # Experience replay storage.
        replay.append((state, action, new_reward, new_state))

        # If we're done observing, start training.
        if t > observe:

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)
            
            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            # WHY RANDOM SAMPLE? COULD TRAINING BE SPED UP BY TAKING LAST BATCHSIZE

            # Get training values.
            X_train, y_train = process_minibatch(minibatch, best_action_model)

            # Train the best_action_model on this batch.
            history = LossHistory()
            best_action_model.fit(
                X_train, y_train, batch_size=batchSize,
                nb_epoch=1, verbose=0, callbacks=[history]
            )
            loss_log.append(history.losses)

        # Update the starting state with S'.
        state = new_state
        cur_speed = new_speed
        cum_rwd += new_reward
        cum_rwd_read += new_rwd_read
        cum_rwd_dist += new_rwd_dist
        cum_rwd_speed += new_rwd_speed
        
        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1/train_frames)
        
        # We died, so update stuff.
        if new_reward == -500 or new_reward == -1000:
            # Log the car's distance at this T.
            data_collect.append([t, car_distance])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\t eps: %f\t dist: %d\t rwd: %d\t read: %d\t dist: %d\t speed: %d\t fps: %d" %
                  (max_car_distance, t, epsilon, car_distance, cum_rwd, \
                   cum_rwd_read, cum_rwd_dist, cum_rwd_speed, int(fps)))

            # Reset.
            car_distance = 0
            cum_rwd = 0
            cum_rwd_read = 0
            cum_rwd_dist = 0
            cum_rwd_speed = 0
            start_time = timeit.default_timer()

        # Save early best_action_model, then every 20,000 frames
        if t % 50000 == 0:
            save_init = False
            best_action_model.save_weights('saved-best_action_models/' + filename + '-' +
                               str(t) + '.h5',
                               overwrite=True)
            print("Saving best_action_model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log)


#def state_frames(new_state, old_state):
#    """
#    Takes a state returned from the game and turns it into a multi-frame state.
#    Create a new array with the new state and first three of old state,
#    which was the previous frame's new state.
#   """
#    # First, turn them back into arrays to make them easy for my small
#    # mind to comprehend.
#    new_state = new_state.tolist()[0]
#    old_state = old_state.tolist()[0][:NUM_SENSORS * (NUM_FRAMES - 1)]
#    # THIS MIGHT BE WHY IT WAS LEARNING SO FAST. W/ FRAMES = 2, IT WAS CONSIDERING TWO MOVES?#
#
#    # Combine them.
#    combined_state = new_state + old_state
#
#    # Re-numpy them on exit.
#    return np.array([combined_state])


def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def process_minibatch(minibatch, best_action_model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our best_action_model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = best_action_model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        newQ = best_action_model.predict(new_state_m, batch_size=1)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, NUM_OUTPUT)) # was 3.
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(NUM_OUTPUT,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])


def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/sonar-frames/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/sonar-frames/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        best_action_model = neural_net(NUM_INPUT, params['nn'])
        train_net(best_action_model, params)
    else:
        print("Already tested.")


if __name__ == "__main__":
    if TUNING:
        param_list = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)

    else:
        nn_param = [240, 160, 80]
        params = {
            "batchSize": 100,
            "buffer": 50000,
            "nn": nn_param
        }
        best_action_model = neural_net(NUM_INPUT, nn_param, NUM_OUTPUT)
        train_net(best_action_model, params)
