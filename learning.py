from flat_game import carmunk
import numpy as np
import random
import csv
from nn import turn_net, speed_net, acquire_net, LossHistory
import os.path
import timeit
import time

# operating modes drive neural net training
HUNT = 1
BEST_TURN = 2
BEST_SPEED = 3
ACQUIRE = 4
cur_mode = ACQUIRE

# turn model settings
TURN_NUM_SENSOR = 5 # front five sonar distance readings
TURN_NUM_OUTPUT = 5 # do nothing, two right turn, two left turn
TURN_NUM_FRAME = 2
TURN_NUM_INPUT = TURN_NUM_FRAME * TURN_NUM_SENSOR

# speed model settings
SPEED_NUM_SENSOR = 3 # front five sonar distance readings, turn action, current speed
SPEED_NUM_OUTPUT = 6 # do nothing, 30, 40, 50, 60, 70
SPEED_NUM_FRAME = 3
SPEED_NUM_INPUT = SPEED_NUM_FRAME * SPEED_NUM_SENSOR

# acquire model settings
ACQUIRE_NUM_SENSOR = 2 # distance, angle
ACQUIRE_NUM_OUTPUT = 5 # nothing, 2 right turn, 2 left turn
ACQUIRE_NUM_FRAME = 2
ACQUIRE_NUM_INPUT = ACQUIRE_NUM_FRAME * ACQUIRE_NUM_SENSOR

# initial settings
START_SPEED = 50
START_TURN_ACTION = 0
START_SPEED_ACTION = 0
START_DISTANCE = 0

# misc
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.


def train_net(turn_model, speed_model, acquire_model, params):
    
    filename = params_to_filename(params)
    
    if cur_mode == BEST_TURN:
        observe = 1000  # Number of frames to observe before training.
    elif cur_mode == BEST_SPEED:
        observe = 2000
    elif cur_mode == ACQUIRE:
        observe = 1000

    epsilon = 1
    train_frames = 700000  # number of flips for training
    batchSize = params['batchSize']
    buffer = params['buffer']

    # initialize variables and structures used below.
    max_car_distance = 0
    car_distance = 0
    t = 0
    cum_rwd = 0
    cum_rwd_read = 0
    cum_rwd_dist = 0
    cum_rwd_speed = 0
    data_collect = []
    replay = []  # stores tuples of (State, Action, Reward, New State').
    save_init = True
    loss_log = []
    oversample = []

    # Create a new game instance.
    game_state = carmunk.GameState()
    
    # Get initial state by doing nothing and getting the state.
    turn_state, speed_state, acquire_state, new_reward, cur_speed, _, _, _ = \
        game_state.frame_step(cur_mode, START_TURN_ACTION, START_SPEED_ACTION,
                              START_SPEED, START_DISTANCE)

    if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
        turn_state = state_frames(turn_state,
                                  np.zeros((1, TURN_NUM_SENSOR * (TURN_NUM_FRAME - 1))),
                                  TURN_NUM_SENSOR, TURN_NUM_FRAME)

    elif cur_mode == ACQUIRE:
        acquire_state = state_frames(acquire_state,
                                   np.zeros((1, ACQUIRE_NUM_SENSOR * (ACQUIRE_NUM_FRAME - 1))),
                                   ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)

    if cur_mode == BEST_SPEED:
        speed_state = state_frames(speed_state,
                                   np.zeros((1, SPEED_NUM_SENSOR * (SPEED_NUM_FRAME - 1))),
                                   SPEED_NUM_SENSOR, SPEED_NUM_FRAME)

    # Let's time it.
    start_time = timeit.default_timer()

    # Run the frames.
    while t < train_frames:
        
        # used to slow things down for de-bugging
        #time.sleep(1)
        
        t += 1 # counts total training distance traveled
        car_distance += 1 # counts distance between crashes

        # Choose an action.
        if random.random() < epsilon or t < observe: # epsilon degrades over flips...
            if cur_mode == BEST_TURN:
                turn_action = np.random.randint(0, TURN_NUM_OUTPUT)
                speed_action = START_SPEED_ACTION
        
            if cur_mode == BEST_SPEED:
                turn_action = np.random.randint(0, TURN_NUM_OUTPUT)
                speed_action = np.random.randint(0, SPEED_NUM_OUTPUT)

            if cur_mode == ACQUIRE:
                turn_action = np.random.randint(0, ACQUIRE_NUM_OUTPUT)
                speed_action = START_SPEED_ACTION

        else: # ...increasing use of predictions over time
            speed_action = START_SPEED_ACTION
            
            if cur_mode in [BEST_TURN, BEST_SPEED]:
            
                turn_qval = turn_model.predict(turn_state, batch_size=1)
            
                turn_action = (np.argmax(turn_qval))  # getting best prediction
            
            elif cur_mode == ACQUIRE:

                acquire_qval = acquire_model.predict(acquire_state, batch_size=1)
    
                turn_action = (np.argmax(acquire_qval))

            if cur_mode == BEST_SPEED:
            
                speed_qval = speed_model.predict(speed_state, batch_size=1)
            
                speed_action = (np.argmax(speed_qval))
        
        # Take action, observe new state and get reward
        new_turn_state, new_speed_state, new_acquire_state, new_reward, new_speed, new_rwd_turn new_rwd_acquire new_rwd_speed = game_state.frame_step(cur_mode, turn_action, speed_action, cur_speed, car_distance)

        # Append (horizontally) historical states for learning speed
        if cur_mode == BEST_TURN or cur_mode == BEST_SPEED:
            new_turn_state = state_frames(new_turn_state, turn_state, TURN_NUM_SENSOR,TURN_NUM_FRAME)
        
        elif cur_mode == ACQUIRE:
            new_acquire_state = state_frames(new_acquire_state, acquire_state, ACQUIRE_NUM_SENSOR, ACQUIRE_NUM_FRAME)
                                      
        if cur_mode == BEST_SPEED:
            new_speed_state = state_frames(new_speed_state, speed_state, SPEED_NUM_SENSOR,SPEED_NUM_FRAME)

        # Experience replay storage.
        if cur_mode == BEST_TURN:
    
            replay.append((turn_state, turn_action, new_reward, new_turn_state))

        elif cur_mode == BEST_SPEED:
            
            replay.append((speed_state, speed_action, new_reward, new_speed_state))

        elif cur_mode == ACQUIRE:
            
            replay.append((acquire_state, turn_action, new_reward, new_acquire_state))

        # If we're done observing, start training.
        if t > observe:
  
            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)
        
            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # Train the turn_model on this batch.
            history = LossHistory()

            if cur_mode == BEST_TURN:
                # Get training values.
                X_train, y_train = process_minibatch(minibatch, turn_model,
                                                     TURN_NUM_INPUT, TURN_NUM_OUTPUT)

                turn_model.fit(X_train, y_train, batch_size=batchSize,
                               nb_epoch=1, verbose=0, callbacks=[history])
            
            elif cur_mode == BEST_SPEED:
                X_train, y_train = process_minibatch(minibatch, speed_model,
                                                     SPEED_NUM_INPUT, SPEED_NUM_OUTPUT)
                
                speed_model.fit(X_train, y_train, batch_size=batchSize,
                               nb_epoch=1, verbose=0, callbacks=[history])

            elif cur_mode == ACQUIRE:
                X_train, y_train = process_minibatch(minibatch, acquire_model,
                                                     ACQUIRE_NUM_INPUT, ACQUIRE_NUM_OUTPUT)
                    
                acquire_model.fit(X_train, y_train, batch_size=batchSize,
                                nb_epoch=1, verbose=0, callbacks=[history])

            loss_log.append(history.losses)

        # Update the starting state with S'.
        turn_state = new_turn_state
        speed_state = new_speed_state
        acquire_state = new_acquire_state
        cur_speed = new_speed
        
        # accumulations
        cum_rwd += new_reward
        cum_rwd_turn += new_rwd_turn
        cum_rwd_acquire += new_rwd_acquire
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
            print("Max: %d at %d\t eps: %f\t dist: %d\t rwd: %d\t read: %d\t newXY: %d\t avg spd: %d\t fps: %d" %
                  (max_car_distance, t, epsilon, car_distance, cum_rwd, \
                   cum_rwd_turn, cum_rwd_acquire, cum_rwd_speed/car_distance, \
                   int(fps)))

            # Reset.
            car_distance = 0
            cum_rwd = 0
            cum_rwd_turn = 0
            cum_rwd_acquire = 0
            cum_rwd_speed = 0
            start_time = timeit.default_timer()

        # oversampling: tripling odds that success record is selected
        #if cur_mode == ACQUIRE and new_reward == 1000:
        #    replay.append((acquire_state, turn_action, new_reward, new_acquire_state))
        #    replay.append((acquire_state, turn_action, new_reward, new_acquire_state))

        if t % 10000 == 0 & car_distance != 0:
            
            print("t: %d\t eps: %f\t dist: %d\t rwd: %d\t read: %d\t comp: %d\t avg spd: %d\t" %
                  (t, epsilon, car_distance, cum_rwd, cum_rwd_turn, cum_rwd_acquire, cum_rwd_speed/car_distance))

        # Save early turn_model, then every 20,000 frames
        if t % 50000 == 0:
            save_init = False
            if cur_mode == BEST_TURN:
                turn_model.save_weights('models/turn/turn-' + filename + '-' +
                                        str(t) + '.h5', overwrite=True)
                print("Saving turn_model %s - %d" % (filename, t))
            
            elif cur_mode == BEST_SPEED:
                speed_model.save_weights('models/speed/speed-' + filename + '-' +
                                         str(t) + '.h5', overwrite=True)
                print("Saving speed_model %s - %d" % (filename, t))
            
            elif cur_mode == ACQUIRE:
                acquire_model.save_weights('models/acquire/acquire-' + filename + '-' +
                                                 str(t) + '.h5', overwrite=True)
                print("Saving acquire_model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    log_results(filename, data_collect, loss_log)


def state_frames(new_state, old_state, num_sensor, num_frame):
    """
    Takes a state returned from the game and turns it into a multi-frame state.
    Create a new array with the new state and first N of old state,
    which was the previous frame's new state.
   """
    # Turn them back into arrays.
    new_state = new_state.tolist()[0]
    old_state = old_state.tolist()[0][:num_sensor * (num_frame - 1)]

    # Combine them.
    combined_state = new_state + old_state

    # Re-numpy them on exit.
    return np.array([combined_state])


def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open('results/sonar-frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/sonar-frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def process_minibatch(minibatch, model, num_input, num_output):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        
        # Get our best move. I think?
        maxQ = np.max(newQ)
        
        y = np.zeros((1, num_output)) # was 3.
        y[:] = old_qval[:]
        
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(num_input,))
        y_train.append(y.reshape(num_output,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def params_to_filename(params):

    if len(params['nn']) == 1:
    
        return str(params['nn'][0]) + '-' + str(params['batchSize']) + \
        '-' + str(params['buffer'])
    
    elif len(params['nn']) == 2:
        
        return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
        str(params['batchSize']) + '-' + str(params['buffer'])

    elif len(params['nn']) == 3:

        return str(params['nn'][0]) + '-' + str(params['nn'][1]) + str(params['nn'][2]) \
        + '-' + str(params['batchSize']) + '-' + str(params['buffer'])


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
        if cur_mode == BEST_TURN:
            turn_model = turn_net(NUM_INPUT, params['nn'])
            train_net(turn_model, 0, params)
        elif cur_mode == BEST_SPEED:
            speed_model = speed_net(NUM_INPUT, params['nn'])
            train_net(0, speed_model, params)
    
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
        if cur_mode == BEST_TURN:
            nn_param = [TURN_NUM_INPUT*25, TURN_NUM_INPUT*10]
            params = {
                "batchSize": 100, # was 100
                "buffer": 50000,
                "nn": nn_param
            }
            turn_model = turn_net(TURN_NUM_INPUT, nn_param, TURN_NUM_OUTPUT)
            speed_model = 0
            acquire_model = 0

        elif cur_mode == BEST_SPEED:
            saved_model = 'models/turn/turn-250-100-100-50000-500000.h5'
            turn_model = turn_net(10, [250, 100], 5, saved_model)
            # need to automate this by extracting model parms via:
            # for layer in model.layers:
                # weights = layer.get_weights()

            nn_param = [SPEED_NUM_INPUT * 25, SPEED_NUM_INPUT * 10]
            params = {
                "batchSize": 100,
                "buffer": 50000,
                "nn": nn_param
            }
            speed_model = speed_net(SPEED_NUM_INPUT, nn_param, SPEED_NUM_OUTPUT)
            acquire_model = 0

        elif cur_mode == ACQUIRE:
            nn_param = [ACQUIRE_NUM_INPUT * 15, ACQUIRE_NUM_INPUT * 5]
            params = {
                "batchSize": 100,
                "buffer": 50000,
                "nn": nn_param
            }
            acquire_model = acquire_net(ACQUIRE_NUM_INPUT, nn_param, ACQUIRE_NUM_OUTPUT)
            speed_model = 0
            turn_model = 0

        train_net(turn_model, speed_model, acquire_model, params)
