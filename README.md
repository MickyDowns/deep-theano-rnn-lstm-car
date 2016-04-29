# Building an intelligent search drone with reinforcement learning

## overview

This project extends work begun by Matt Harvey in his Github repository entitled **Using reinforcement learning to train an autonomous vehicle to avoid obstacles** located here: https://github.com/harvitronix/reinforcement-learning-car. He trained an digital autonomous car to avoid obstacles using reinforcement learning. His ultimate goal is to embed the resulting neural network onto a chip and into a a car that will, hopefully, avoid his cats while cruising his house. I highly recommend his Medium posts on the topic beginning here: https://medium.com/@harvitronix/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6. They were instrumental to my understanding of reinforcement learning and Q-Learning.

My challenge is similar to Matt's. However, I am building a drone. Aside from the obvious challenge of working in 3 dimensions, this drone must be able to perform some reasonably complex tasks. Specifically, I want it to search, locate and report the position of specific objects in its environment. It may be possible to perform these tasks using a deep, deep neural net. The folks at DeepMind (of Go fame) used Q-learning to train networks to win Atari games (https://t.co/liV9sJFoCp). However, there are two limitations to that approach from my perspective. First, providing a convnet a degraded (down-sampled) view of the entire environment is not realistic given my expected end-state environment (think of drones in houses, warehouses and caves). This drone needs to figure out it's environment by itself. Second, deep reinforcement networks often require tons of GPUs to train. I want a net that is capable of running (and training) on CPUs. Why? Because I want the drone itself to be capable of low-level learning and it's easier/cheaper to put a CPU in a drone than an expensive GPU.

This project is divided into three sections. This first section deals with training the base neural nets to work together in 2-space to perform a skeleton of the final objective. The second section will be to covert those models to work in three dimensions. The final section will be to embed the algorithms in a chip, on a drone. 

The project technology is Python3, Pygame, Pymunk, Keras and Theano. See "installing" below to get started though be ready to do some problem solving as, given differences in environments, components like Pymunk and Pygame can be problematic. Further, working with Pygame is, frankly, challenging. So, I'll be specific about the process of traning the models and the results. So, even if you don't install and run the code, you'll understand the approach.

## part 1: building 2-dimensional networks that work together

The first, 2-D, implementation of the models is accomplished through four, independently-trained neural networks in three categories of operation. They are:

* avoid: A drone that is not in the air, because it hit a tree, a wall or a person isn't of much use. So, the first two neural networks enable the drone to avoid obstacles. 

    ** The first network trains the drone, that is moving at constant speed, to avoid both stationary and moving obstacles in, or headed into, its path. The input to this "turn" "neural network is a set of five sonar sensor readings the eminate from the front of the drone. The outputs are five turn actions including: going straight, turning right (2 levels), and turning left (2 levels). Its objective function is to avoid collisions by maximizing the length of its sensor readings. It receives mini rewards each time it does and looses rewards each time it crashes into an obstacle or a wall.

    ** The second network trains the drone to speed up or slow down to avoid obstacles. This "speed" "network is trained on-top of the "turn" network. That is, its input is the minimum distance measured by the afformentioned sonar sensors AND the turn decision of the "turn" model. So, it's learning about distances and speeds thru the experience of the lower model. It's output is one of five speed options: 30, 40, 50, 60, 70. Note: the default speed is 50. It's reward function is the 3-dimensional composite built on the two dimensions of speed and distance to obstacles picuted here:. If that's confusing, just think it's rewarded for going slow when objects are close and fast when objects are far.'

    ** In the second part, when the challenge becomes 3-dimensional, I'll add a "height" model. Like the "speed" model it will be trained on top of the "turn" model.

* search:

    ** ??In the third part, when the challenge becomes real-world, I'll ad a "boundaries" model. It will be responsible for keeping the drone w/in a set of gps boundaries and to foce the drone to land outside thos boundaries.?? OR do you want it to self-identify the search area based on xtics and the boundaries based on distance from the origin. 

        Note: realistically, this can be any arbitrary set of contditions so long as the result can be quantified and presented to the drone. 

    ** The third network trains the drone to search a pre-defined area. Specifically, it is given a 1000x700 pixel total area and an 800x500 target search area. Its challenge is to locate the target search area and "touch" goals (pixels) dispersed over the search area to recieve rewards. Its input are the "coordinates" (each represented by a single pixel id serially asigned from the origin) of bonties and the status of bounties (0 for available, 1 for taken). It has three outputs: straight, left, right and a constant speed (25. )
        
        Note: This approach decreases, but does not eliminate crashes. This model has no memory of wall locations, nore does it have  as the walls are not provided and it has no way of detecting them. However, to the extent it is successful in focusing it's steps on the center bounty area, crashes should decrease.'

* synthecize:

    ** The fifth network pulls it all together selectively employing the four lower level networks to achieve the overall objective (searching the target area) while avoiding obstacles. The deepest network, it has x layers. It's inputs are: the turn instruction from the search network, the turn and speed instructions from the avoid networks, the distance to the nearest obstacle and the distance to the nearest goal. Its sole role is to select  instruction from the

## how it's implemented

four key files. 

* carmunk handles game play. learning is link between game and net. provided imputs to and receives outputs from each (game play/carmunk). ...

## key learnings

The coursework I've taken in statistical and computational methods have helped. However, in academic cases the dots are often pre-connected... you knew a positive outcome was possible. It was just a matter of figuring out how. That's not the case here. That gap was instructive for me. So, I'll pass along some of the key learnings:

* importance of tight linkage between inputs (what the model sees/hears and is, therefore, capable of responding to), outputs (the levers the model controls to improve its performance), the objecive function, and the cost function (the deviation from the objective function). Think like a machine. What are you seeing. 

* complexity breakthrough by choosing deeper network

* learning gain using frames

map its environment, develop an search plan, then search its environment looking for specific objects. the environment using its camer  to search that environment, then to search the Further, I wanted the drone  had three distinct objectives. 'This project uses  



## to run for your first time

### installing

1. Clone this repo
1. Install numpy ```pip3 install numpy```
2. Install Pygame. I used these instructions: http://askubuntu.com/questions/401342/how-to-download-pygame-in-python3-3 but with ```pip3 install hg+http://bitbucket.org/pygame/pygame``` after I installed the dependencies
3. Install pymunk ```pip3 install pymunk```
4. Update pymunk to python3 by CDing into its directory and running ```2to3 -w *.py```
5. Install Keras ```pip3 install keras```
6. Upgrade Theanos ```pip3 install git+git://github.com/Theano/Theano.git --upgrade --no-deps```
7. Install h5py for saving models ```pip3 install h5py```

### Training

First, you need to train a model. This will save weights to the `saved-models` folder. You can do this by running:

`python3 learning.py`

It can take anywhere from an hour to 36 hours to train a model, depending on the complexity of the network and the size of your sample. However, it will spit out weights every 25,000 frames, so you can move on to the next step in much less time.

### Playing

Edit the `nn.py` file to change the path name for the model you want to load. Sorry about this, I know it should be a command line argument.

Then, watch the car drive itself around the obstacles!

`python3 playing.py`

That's all there is to it.

### plotting

Once you have a bunch of CSV files created via the learning, you can convert those into graphs by running:

`python3 plotting.py`

This will also spit out a bunch of loss and distance averages at the different parameters.

## Credits

I'm grateful to the following people and the work they did that helped me learn how to do this:

- Deep learning to play Atari games: https://github.com/spragunr/deep_q_rl
- Another deep learning project for video games: https://github.com/asrivat1/DeepLearningVideoGames
- A great tutorial on reinforcement learning that a lot of my project is based on: http://outlace.com/Reinforcement-Learning-Part-3/
