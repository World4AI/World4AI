========================
States, Actions, Rewards
========================

Interaction
===========

The agent and the environment interact continuously, each reacting to the data sent by the other. 

.. note::
    In reinforcement learning the sequential information flow between the agent and the environment is called **interaction**.


.. figure:: ../_static/images/reinforcement_learning/interaction/interaction.svg
   :align: center
    
   Interaction between the agent and the environment.


In essence interaction means that there is a communication channel between the agent and the environment, where data flows sequentially between the two. 

In order for the data to be communicated there have to be three consecutive steps.

#. The agent / environment receives the data
#. The data is processed by the receiver
#. The processed data is sent back to the originator

.. figure:: ../_static/images/reinforcement_learning/interaction/sequential.svg
   :align: center
    
   Sequential flow of data in reinforcement learning.

What is important to mention is that this stream of data is sent in a strictly sequential way. When the environment sends the data for example, it has to wait until it receives the response from the agent. Only then a new batch of data can be sent to the agent again.  

The middle part of the three steps, the processing of the data by the **agent** is the main part of reinforcement learning. The processing will be covered in detail in later chapters. In this chapter I will only describe the types of data. 

In general in reinforcement learning there are just 3 types of data that need to be sent between the agent and the environment: **state** data, **action** data and **reward** data. 

.. figure:: ../_static/images/reinforcement_learning/interaction/state_action_reward.svg
   :align: center
    
   State, action, reward.

The agent receives the current state of the environment and based on the state sends the action he would like to take. The environment sends the reward for that action and transitions into a new state taking the action of the agent and the current state into account. The agent then uses the reward to learn to make decisions based on states. 

.. note::
   The state, the action and the reward all have to be numerical values, in order for the computer to process them.

Math Sidenote
=============

Following are some mathematical definitions to be able to follow the following section.

.. note:: 
   Discrete variables are integers like :math:`1, 2, 3, 42`

   Continuous Variables are floating point numbers like :math:`3.14, 22.2, 14.992`


.. note::
   A scalar is a single number like :math:`3.14`.

   A vector is a one-dimensional collection of numbers.

   .. math::
      \begin{bmatrix}
      1\\
      4
      \end{bmatrix} 

   A matrix is a two-dimensional collection of numbers.

   .. math::
      \begin{bmatrix}
      1 & 2 & 3\\
      4 & 5 & 6
      \end{bmatrix}

   A tensor is a multi-dimensional collection of numbers.


State
=====

.. note::
   The state is the representation of the current condition of the environment.
 
The state describes how the environment actually looks like. It is the current situation the agent faces and based on the state the agent has to make his decisions. The state can be represented by a scalar, a vector, a matrix or a tensor and can be either discrete or continuous.

.. figure:: ../_static/images/reinforcement_learning/interaction/state.svg
   :align: center
    
   State in the gridworld.

In this simple gridworld example all the agent needs to know to make the decisions is the location of the circle in the environment. So in the starting position the state would be 1. The one to the right 2 and so on. Based on the position the agent can choose the path towards the triangle. 

There are of course more complex environments where the state is for example represented by a tensor containing rgb values. [#]_

Action
======

.. note::
   The action is the representation of the decision/behaviour of the agent.

The action is the behaviour the agent chooses based on the state of the environment. [#]_ Like the state the action can be a scalar, a vector, a matrix or a tensor of discrete or continuous values. 

.. figure:: ../_static/images/reinforcement_learning/interaction/action.svg
   :align: center
    
   Action in the gridworld.

In the above gridworld example the agent can move north, east, south and west. Each action is encoded by a discrete scalar value.

* North = 1
* East = 2
* South = 3
* West = 4

One of the above scalar values is sent back to the environment.
 
Reward
======

.. note::
   The reward is the signal to reinforce certain behaviour of the agent to achieve the goal of the environment. 

The reward is what the agent receives from the environment for an action. It is the value that the environment uses to reinforce a behaviour to solve an environment and it is the value that the agent uses to improve his behaviour. 

Unlike the action or the state the reward has to be a scalar, one single number, it is not possible for the reward to be a vector, matrix or tensor. As expected larger numbers represent larger or better rewards so that the reward of 1 is higher than the reward of -1. 

.. figure:: ../_static/images/reinforcement_learning/interaction/reward.svg
   :align: center
    
   Reward in the gridworld.

In this gridworld example the agent receives a reward of -1 for each step taken with the exception of taking a step towards the triangle, where the agent receives a reward of 1. 
   

Timestep
========

.. note:: In reinforcement learning each iteration of exchanging state+reward and action is called a timestep.

Reinforcement learning works in (mostly) discrete timesteps. Each iteration where the environment and the agent each have sent their data constitutes a timestep.

Notes
=====

.. [#] RGB stands for red, green, blue and is a common way to represent images. 
.. [#] Theoretically the agent can make random decisions, but to maximize the sum of rewards agents should base decisions on the state.
