====================================
Properties of Reinforcement Learning
====================================

There are probably dozens of formal definitions of reinforcement learning. These definitions do not necessarily contradict each other, but rather explain something similar when I look a little deeper at what the definitions are trying to convey. I will introduce the one definition that in my opinion captures the essence of reinforcement learning in a very clear and succinct way.

.. note:: 
          Reinforcement Learning is **Learning** through **Trial and Error** and **Delayed** **Rewards**. [#]_

The definition consists of three distinct parts: *Learning*, *Trial and Error* and *Delayed Rewards*. In order to understand the definition I will deconstruct the sentence and look at each part individually. 

Learning
========

Learning is probably the most obvious part of the definition. Usually in reinforcement learning when the agent starts to interact with the environment the agent does not know anything about that environment. The assumption in reinforcement learning that is always made is that the environment the agent interacts with contains some goal that the agent has to achieve. 

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/grid_world.svg
   :align: center

   In this grid world the goal is for the circle to reach the triangle.

For example the agent is expected to move the circle from the starting cell position (bottom left corner) to the goal cell position (top left corner). 

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/learning.svg
   :align: center

   Suboptimal (red) and optimal (green) strategy.

When I talk about learning, usually that means that the agent gets better at achieving that particular goal over time. He could start by moving in a random fashion (as indicated by the red arrows) and over time learn the best possible (meaning the shortest) route (as indicated by the green arrows). 

.. note:: 
   For the agent learning means getting better at achieving the goal of the environment.

   

Rewards
=======

The question still remains how exactly does the agent know what the goal of the environment actually is? The environment with which the agent interacts gives feedback about the behaviour of the agent by giving out a reward after each single step that the agent takes. [#]_

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/rewards.svg
   :align: center

   Negative (red) and positive (green) rewards.

If the goal of the grid world is to move the circle to the cell with the triangle as fast as possible the environment could for example give positive reward for getting to the cell with the triangle and punish the agent in any other case. 

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/routes.svg
   :align: center

   Different paths in the environment generate different sum of rewards.

If the agent takes the direct route (path 1) to the triangle he will get less negative rewards while an indirect route creates a lot of negative rewards. The agent needs to learn through the reward feedback that some sequences of actions are better than others. 

.. note:: 
   In reinforcement learning the agent learns to maximize the sum of rewards. The goal of the environment is implicitly contained in the rewards.

Trial and Error
===============

The problem with the rewards is that it is not clear from the very beginning what path produces the highest possible sum of rewards. In reinforcement learning there is only the reward signal and even if the agent receives a positive reward you never know if he could have done better. Unlike in supervised learning, there is no teacher/supervisor to tell the agent what the best behaviour is. So how can the agent figure out what sequence of actions produces the highest sum of rewards? The only way he can. By trial and error.

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/trial_error.svg
   :align: center

   The agent tries out different behaviour.

The agent has to try out different behaviour to figure out which one produces optimal results. How long it takes the agent to find a good sequence of decisions depends on the complexity of the environment and the employed learning algorithm. It can be anything between a couple of seconds to many days. In some cases we can not solve an environment no matter how hard we try. 

.. note:: 
   In the context of reinforcement learning, trial and error means trying out different sequences of decisions and comparing the resulting sum of rewards. 

Delayed
=======

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/delayed.svg
   :align: center

   In reinforcement learning rewards are often delayed.

In reinforcement learning the agent often needs to take dozens or even thousands of steps before a particular reward is achieved. In that case there has been a succession of many steps and the agent has to decide which step and in which proportion is responsible for the reward, so that the agent could select the decisions that lead to a good sequence of rewards more often. 

.. figure:: ../../_static/images/reinforcement_learning/intuition/properties_of_rl/credit_assignment.svg
   :align: center

   The credit assignment problem.

Which of the steps is responsible for the positive reward in the image above? Is it the action just prior to the reward? Or the one before that? Or the one before that? Reinforcement Learning has no easy answer to the question which decision gets the credit for the reward. This problem is called *the credit assignment problem*. 

.. note::
   In reinforcement learning rewards for an action are often delayed, which leads to the credit assignment problem. 


Notes
=====

.. [#] This definition is highly inspired by the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto.
.. [#] In reinforcement learning we do not actually differentiate between a reward and a punishment. We call it reward no matter if the reward is positive, negative or zero. 