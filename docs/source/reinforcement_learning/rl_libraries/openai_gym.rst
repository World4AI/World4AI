==========
OpenAI Gym
==========

Introduction
============

OpenAI Gym is the de facto official collection of environments for the research community. Whenever you read a research paper describing the results for common reinforcement learning tasks, chances are high that OpenAI Gym was used. Gym contains several categories of environments ranging from easy discrete toy text problems like "Frozen Lake" and 2d physics based environments like “Lunar Lander” to complex 3d physics MuJoCo environments like "Ant" and "Hopper". The list and a short description for each of the environments can be found on the official website
https://gym.openai.com/envs.

The beauty of the library is a common interface for all the environments in the library, meaning that you can implement a single algorithm and apply it to many different environments. For example the same algorithm can be tested against every single Atari game. More than that, Gym allows the creation of custom made environments. The general rule is that every imaginable environment can be created, as long as these implement some base classes of Gym and adhere to the interface standards. 

Speaking in slightly broader terms OpenAI Gym is a library that contains many commonly used MDPs and allows the creation of new custom MDPs. The knowledge about the agent-environment interface is general and is applicable to all environments.

OpenAI Gym contains only single-agent environments. For multi-agent environments other libraries have to be used.

Installation
============

The installation process is quite straightforward. Create a virtual environment using either conda or venv and install the packages either with the conda install or pip install commands. 

.. code::

    pip install gym
    pip install Box2D
    pip install gym[atari]

After the installation the library can be imported

.. code:: python

    import gym


Initialization
==============

The initialization of the environment is a one liner.

.. code:: python
   
    env = gym.make('FrozenLake-v0')

The function creates and returns an instance of the Env (environment) class. The only adjustment required to load a different environment is to change the string representation that is given as the parameter to the function.

>>> import gym
>>> env = gym.make('FrozenLake-v0')
>>> for base_class in type(env).mro():
...     print(base_class)
...
<class 'gym.wrappers.time_limit.TimeLimit'>
<class 'gym.core.Wrapper'>
<class 'gym.core.Env'>
<class 'object'>

There are several derived classes for environments, but at the base of all of them is the gym.core.Env class. 

Spaces
======

Space is the base class that is used to implement action and observation/state spaces. The Space class itself is never used to instantiate the action/observation spaces, but provides basic functionalities that are inherited by all space implementations. The actual implementations are delegated to the derived classes discussed below.

Three methods are of interest when working with OpenAI Gym Space.

* The sample() method samples a random Action or Observation from the set of available Actions or Observations.
* The seed(seed=None) method seeds the pseudo random number generator. This method is especially important if you intend to use the sample() method and want to create reproducible results.
* The contains(x) method takes an argument and checks it is contained in the set. 

To get the action or the observation space of the environment the following code can be used.

>>> import gym
>>> env = gym.make("FrozenLake-v0")
>>> action_space = env.action_space
>>> observation_space = env.observation_space

Discrete
--------

The Discrete class derives from the Space class and is intended for discrete observation and action spaces. Additionally to the methods and properties derived from Space the Discrete implements the n property. The n property indicates the number of items in the set. The items contained in the set start with 0 and end with n-1. For example if n = 5 for the observation set then the set is :math:`S = {0, 1, 2, 3, 4}`.

>>> import gym
>>> env = gym.make("FrozenLake-v0")
>>> action_space = env.action_space
>>> observation_space = env.observation_space
>>> type(action_space)
<class 'gym.spaces.discrete.Discrete'>
>>> type(observation_space)
<class 'gym.spaces.discrete.Discrete'>
>>> action_space
Discrete(4)
>>> observation_space
Discrete(16)
>>> action_space.n
4
>>> observation_space.n
16
>>>

The “Frozen Lake” environment for example has a discrete action and observation space. There are 4 actions :math:`A = {0, 1, 2, 3}` to move in 4 directions and 16 states :math:`S = {0, 1, 2, … , 14, 15}`.


Box
---

Similar to the Discrete class the Box class inherits from the Space class. Box is intended to deal with continuous environments and provides specific properties and methods for that purpose. 

* The shape property returns the shape/dimensionality of the space
* The low property returns a list of lower boundaries for the space
* The high property returns a list of higher boundaries for the space
* The dtype property returns the numerical type of the space


>>> import gym
>>> env = gym.make('CartPole-v1')
>>> action_space = env.action_space
>>> observation_space = env.observation_space
>>> type(action_space)
<class 'gym.spaces.discrete.Discrete'>
>>> type(observation_space)
<class 'gym.spaces.box.Box'>
>>> action_space
Discrete(2)
>>> observation_space
Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
>>> action_space.n
2
>>> observation_space.low
array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38],
      dtype=float32)
>>> observation_space.high
array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38],
      dtype=float32)
>>> observation_space.shape
(4,)
>>> observation_space.dtype
dtype('float32')


The Environment Interface
=========================

The environment provides several methods to account for the agent-environment interaction expected from the MDP setting. 

The reset() method resets the environment and returns an initial observation of the environment. The method has to be called either to initialize the environment for the first time or to reinitialize the environment after a terminal state has been reached. 
 
The step() method takes the action from the agent as the argument and returns a tuple containing 4 elements

* the next observation 
* the reward 
* done,  a boolean flag indicating if the environment reached the terminal state 
* info,  information used for debugging

.. code:: python

    import gym

    env = gym.make('FrozenLake-v0')

    # initialize the variables
    done, obs = False, env.reset()

    # interaction loop
    while not done:
        # agent selects an action
        # here the decision is done randomly, but usually it is based on the obs variable
        action = env.action_space.sample()
        # 1. agent takes the action
        # 2. the environment transitions into the next state
        # 3. the next observation, the reward and the done flag are send back to the agent
        next_obs, reward, done, info = env.step(action)
        
        # the obs variable is reset in preparation for the next episode
        obs = next_obs
    

The render() method renders the environment on the screen.

The close() method closes the environment.

The seed(seed=None), the seed is useful if you want reproducible results


Wrappers
========

Coming Soon

Monitor
=======

Coming Soon
