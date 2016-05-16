# Pack of Drones: Layered reinforcement learning for complex behaviors

## overview

I'm teaching my daughters about artificial intelligence. This project introduces key concepts around reinforcement learning, using sensor data, writing objective functions and layering networks. The result, to this point, is a 2-D implementation of learning drones that:

    *   Avoid: Turn, accelerate, decelerate to avoid obstacles

    *   Acquire: Locate and touch a target object (pixels) in the environment

    *   Hunt: Acquire target objects while avoiding obstacles

    *   (IN PROCESS) Pack: Accelerate and improve hunt results by directing >1 drones

Here's a video overview: [![layered reinforcement learning for complex behaviors](https://img.youtube.com/vi/WrLRGzbfeZc/0.jpg)](https://www.youtube.com/watch?v=WrLRGzbfeZc)

In general, layered learning is used when mapping directly from inputs to outputs is not tractable e.g., too complex for current algorithms, networks. Layered learning starts with a bottoms-up, hierarchical task decomposition. It then uses machine learning algorithms to exploit data at each level to train specific, separable models. Finally, the output of learning in one layer feeds into the next layer until the complex behavior is achieved. More here: http://www.cs.cmu.edu/~mmv/papers/00ecml-llearning.pdf. 

It is possible that the tasks performed herein could be performed by a single, deep, deep neural net. The folks at DeepMind (of Go fame) used Q-learning to train a single network to win Atari games (https://t.co/liV9sJFoCp). However, I have specifically avoided giving the drones a down-sampled, convnet view of the entire environment. Instead, they learn it thru exploration as I will eventually flash this network to real drones and want the learning to continue onboard, on CPU. 

This project extends work begun by Matt Harvey in his Github repository entitled **Using reinforcement learning to train an autonomous vehicle to avoid obstacles** located here: https://github.com/harvitronix/reinforcement-learning-car. He trained an digital autonomous car to avoid obstacles using reinforcement learning. His ultimate goal is to embed the resulting neural network onto a chip and into a a car that will, hopefully, avoid his cats while cruising his house. I highly recommend his Medium posts on the topic beginning here: https://medium.com/@harvitronix/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6.

## structure

The project technology is Python3, Pygame, Pymunk, Keras and Theano. See "installing" below to get started though be ready to do some problem solving as, given differences in environments, components like Pymunk and Pygame can be problematic. Further, working with Pygame is challenging. I've extensively commented the code. So, even if you don't install and run the code, you'll understand the approach.

### networks that work together in 2-D

The first, 2-D, implementation of the models is accomplished through four, independently-trained neural networks in three categories of operation. They are:

* avoid: A drone that is not in the air, because it hit a tree, a wall or a person, isn't of much use. So, the first two neural networks enable the drone to avoid obstacles. 

    ** The first network trains a drone, that is moving at constant speed, to avoid stationary and moving obstacles. The input to this "turn" "neural network is a set of five sonar sensor readings that emanate from the front of the drone. The outputs are turn actions including: going straight, turning right (2 levels), and turning left (2 levels). Its objective function is to avoid collisions by maximizing the length of its sensor readings. It receives rewards for long readings and penalties for short readings (crashes).

    ** The second network trains the drone to speed up or slow down to avoid obstacles. This "speed" "network is trained on-top of the "turn" network. That is, its input is the minimum distance measured by the sonar sensors AND the turn decision of the "turn" model. So, it's learning about object distances and speeds in the context of turn decisions made by the lower model. It's output is one of four speed options: 0, 30, 50, 70. It's trained using a 3-dimensional objective function blending speed and distance from obstacles. Basically, it's rewarded for slowing down in traffic and speeding up on the open road.'

* acquire: 

    ** The third network trains the drone to search a pre-defined surface (the 1000x700 pixel board) to locate and "touch" a target pixel. Its inputs are the target pixel "coordinates" i.e., a heading adjustment given the drone's current direction and distance. So, when it sees (-180, 150), it learns to turn around (until heading = 0), then travel forward 150 pixels. It has the same five "turn" model outputs (straight, 2 right, 2 left). Finally, it is rewarded for moving efficiently to and then acquiring the pixel. It is penalized for moving away from the pixel or crashing into the walls. Note: This approach decreases, but does not eliminate crashes. This model has no distance sensors and no memory of wall locations. However, to the extent it is successful in focusing its path to the target in a straight, tight line, it avoids walls.

* hunt:

    ** The fourth network selectively employs the three lower level networks to achieve a complex "hunting" objective. It searches the target area acquiring target pixels while avoiding obstacles. Its inputs are the full array of sonar sensor readings (distance and object color), heading to target and distance to target. Its outputs are either the "acquire" or "avoid" networks (recall "avoid" is in turn a composite of "turn" and "speed" networks). It selectively accepts one or the other's suggested action based on its appraisal of the inputs and its objective function. That objective function is a normalized composite of defensive (avoiding obstacles) and offensive (acquiring target pixels) move efficiency. Note: This model developed reasonably complex hunting behaviors incuding circling the target pixel to find the best angle of attack and a waiting for obstacles to clear its path before proceeding. 

### key files

* learning.py: Trains the neural network based based on predictions/decisions, states and rewards. While it contains code to train each of the four networks, they are all trained in similar fashion. Some number of random training samples are generated and evaluated against the objective function. A "state" (of the game world) is generated for each sample move and its efficacy is evaluated against the objective function for that network. These states are stacked (up to 100k), sampled and the submitted in microbatches of 100 to the nets for training. Gradually, they improve after seeing a variety of states with corresponding rewards.

* carmunk.py: Controls the game play. It receives drone turn and speed commands from learning.py and playing.py. It effects those moves in the game and returns a set of training mode-specific (e.g., turn, speed, acquire, hunt) states. Those states communicate key information about obstacle distances, crashes, rewards, etc. used in training the models. Look in this file if you want to know how the drone and obstacles work, how state is maintained and how rewards are calculated.

* nn.py: Holds the network schema.

* playing.py: Runs the trained neural networks receiving states from carmunk.py, getting predictions (turn and speed actions) using nn.py returning those to the game, etc.

## learnings

Coursework in statistical and computational methods helped here. However, in academic cases the dots are often pre-connected... you know a positive outcome is possible. That's not the case here. That gap was instructive for me. So, I'll pass along some of the key learnings:

* Importance of tight linkage between inputs, outputs, objecive function, and cost function. Seems obvious to say that the networks can only learn from stimuli they can observe, can only change levers they control, and can only improve if they understand what's valuable. So, when designing the network for a layer **think like a machine**. Would you know what lever to pull based on the incoming data stream and a numeric reward?

* **Frames speed learning.** The current networks do are not using LSTM (Long Short Term Memory). LSTM was specifically developed to enable networks to learn from prior experience. The networks used here have short term memory in that each prediction is based on joining states from the last two or three moves. This roughly halved training time. 

* Run the **cost-benefit on network depth**. Initial "speed" and "hunt" models using two-hidden layers couldn't handle the complexity of a. reading sensors, b. reading lower model decisions, c. deciphering 3-D objective functions. Performance declined. Adding a third hidden layer enabled these models (at some cost to processing speed) to improve survivability. In contrast, 3 hidden layer "turn" and "acquire" models overfit the data. Increasing regularization in these models by a. dropping hidden layers and b. introducing dropout, improved survivability. 

* **Use domain experience.** In a layered network like this the simplier the objective function at each layer the better chance you'll have of the layers working together. However, that's not always the case. The "speed" model was implemented first with a simple survival goal (i.e., same as turn: reward = min(sensor_lengths)), second with a more complex 3D goal for slowing in traffic. Given training < 700k frames, the more complex objective function significantly outperformed (2x) the simplier. 


## to run for your first time

### installing

1. Clone this repo
1. Install numpy ```pip3 install numpy```
2. Install Pygame. I used these instructions: http://askubuntu.com/questions/401342/how-to-download-pygame-in-python3-3 but with ```pip3 install hg+http://bitbucket.org/pygame/pygame```
3. Install pymunk ```pip3 install pymunk```
4. Update pymunk to python3 by CDing into its directory and running ```2to3 -w *.py```
5. Install Keras ```pip3 install keras```
6. Upgrade Theanos ```pip3 install git+git://github.com/Theano/Theano.git --upgrade --no-deps```
7. Install h5py for saving models ```pip3 install h5py```