=======================================
Definition of a Markov Decision Process
=======================================

Motivation
==========

Many of the components of a Markov Decision Process (MDP) were already introduced in one of the previous chapters. The main focus of those chapters was the building of an intuitive foundation. In the following chapters I am going to reiterate the already introduced material, but the focus is going to be on the more formal explanations of MDPs. The mathematics that is going to be introduced in the next chapters will form the basis of much of reinforcement learning.

.. note::
   A Markov Decision Process (MPD) is a formal description of a sequential decision problem with uncertainty.

In essence an MDP allows us to formalize the interaction loop between the agent and the environment, where the actions of the agent influence future states/rewards and the agent might have to decide to forego the current reward to get higher rewards in the future. The common assumption in reinforcement learning is the existence of an MDP at the core of each environment.

HERE IS AN IMAGE

The interaction is done sequentially, where the agent and the environment take turns to react to each other. Each iteration of actions, rewards and states happens in a period of time, called a time step, :math:`t`. The time step is a discrete variable starting at 0 and increasing by 1 after each iteration. During the first time step the agent receives the initial state of the environment :math:`S_0` and reacts accordingly with the action :math:`A_0`. The environment transitions into a new state :math:`S_1` and generates the reward :math:`R_1`. The agent in turn reacts with the action :math:`A_1` and the interaction continues. The general notation of writing States, Actions and Rewards is :math:`S_t, A_t, R_t` where the subscript :math:`t` represents a particular time step.

From a Stochastic Process to an MDP
===================================

A markov decision process consists of three parts. It involves a stochastic **process**, it abides by the **markov** property and there is the possibility to influence the states through **decisions**.

Stochastic Process
------------------

.. note:: 
   A stochastic or random process can be defined as a sequence of random variables.

HERE IS AN IMAGE

In the above image there are no actions or rewards yet and the state evolves randomly over time following a stochastic process. There are two distinct states that the process can be in, the 1 and the 0. Each of the states can be reached  with 50% probability and the new state does not depend on any previous states. In essence this random process corresponds to a sequence of coin tosses where for example heads would correspond to 0 and tails to 1. 

For the above process the following can be said: :math:`Pr(S_{t+1} \mid S_t) = Pr(S_{t+1})`

:math:`Pr(S_{t+1})` is the probability that a certain state will be tossed, in the above case :math:`Pr(S_{t+1}=HEADS) = 0.5` and :math:`Pr(S_{t+1}=TAILS)=0.5`. :math:`Pr(S_{t+1} \mid S_t)`, reads as x given y, depicts a conditional probability, where the probability of being in the new state :math:`S_{t+1}` depends on the current state :math:`S_t`. For example :math:`Pr(S_{t+1}=HEADS|S_t=TAILS)` shows the probability of a coin toss having a value of HEADS when the previous toss had a value of TAILS. When you consider a coin toss, then the new occurrence of either heads or tails does not depend on the previous specific value of the toss. The events are independent. :math:`Pr(S_{t+1} \mid S_t) = Pr(S_t)` means that knowing the last value of a coin toss does not give us any more knowledge regarding the future toss. :math:`Pr(S_{t+1} \mid S_t) = Pr(S_t) = 0.5`.

.. list-table:: Bernouli Process

   * - Process
     - yes
   * - Markov
     - no
   * - Decisions
     - no

Markov Chain
------------

.. note::
   A Markov chain is a stochastic process that has the Markov property. 
    
   Markov Property: :math:`Pr[S_{t+1} \mid S_t] = Pr[S_{t+1} \mid S_1, .... , S_t]` 
    
   The Markov property, or memorylessness, means that the next state only depends on the current state and not the states before that. 
    
HERE IS AN IMAGE

Unlike in the coin toss example, in a Markov chain the probability to be in the state :math:`S_{t+1}` depends on previous states, but only the most recent state, :math:`S_t`, is relevant. The above image shows that each state depends on the previous state, while the previous information, the states that came before that, is irrelevant. The markov property is extremely convenient, as only the most recent events need to be tracked, which allows for more tractable computations.

HERE IS AN IMAGE

Each of the color coded circles above represents a state, while the numeric values near the arrows represent the transition probabilities from one state to another state. For example...

HERE IS AN IMAGE

If we unfold the process and follow it for a while, the following sequence might for example occur. 

.. math::
   S_0, S_1, S_2, S_3, S_4, ..., S_t

.. list-table:: Markov Chain

   * - Process
     - yes
   * - Markov
     - yes
   * - Decisions
     - no

MDP
---

HERE IS AN IMAGE

A Markov chain can be extended to a Markov Decision Process with the introduction of rewards and actions. While in the case of a Markov chain the states evolve without any possibility of an influence on the environment, in the case of an MDP the agent has “agency” over his actions and gets rewards for his behaviour.

The unrolled MDP forms a sequence of States, Actions and Rewards, called a trajectory.

.. math::
   S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, ...

.. list-table:: MDP

   * - Process
     - yes
   * - Markov
     - yes
   * - Decisions
     - yes


.. note::
   A tuple is a finite ordered list of elements
  

In more mathematical terms a Markov decision process is a 5-tuple, :math:`(\mathcal{S, A}, P, R, \gamma)`.
   
Frozen Lake
***********

HERE IS AN IMAGE

To explain the contents of the tuple I will introduce a new environment. “Frozen Lake” is a beginner level environment, suited well to explain the underlying components of an MDP. As the name of the environment suggests, the surface of the lake is frozen. This causes the surface to be either safe, but slippery or not safe at all. The player starts at the top left corner (indicated by the letter S as in Start). The goal of the environment is to reach the bottom right corner (indicated by the letter G as in Goal). The safe surface comprises the F (F as in Frozen) and the G cells. The unsafe surface is indicated by the H (H as in Hole) cells. The reward is in most cases 0, unless the agent reaches the goal where he achieves a reward of 1. The agent can move in 4 directions. When the agent tries to move into the direction of the wall the resulting state is the same as the previous state. The environment itself is stochastic. When the agent chooses an action in ⅓ of the cases the environment moves the player in that direction, while in ⅔ of the cases the player is moved into one of the orthogonal directions (divided equally). 

HERE IS AN IMAGE

:math:`\mathcal{S}`
*******************

.. note:: 
   :math:`\mathcal{S}` is the set of all legal states


HERE IS AN IMAGE

HERE IS AN EXPLANATION

   
:math:`\mathcal{A}`
*******************
.. note::
   :math:`\mathcal{A}` is the set of all legal actions

HERE IS AN IMAGE

HERE IS AN EXPLANATION

:math:`P`
*********

.. note:: 
   :math:`P` is the transition model. 

   :math:`P(s' \mid s, a) \doteq Pr[S_{t+1}=s' \mid S_t=s, A_t=a]`

   The transition model is the function  that calculates the probability of landing in some state :math:`s'` at timestep :math:`t+1` when at timestep :math:`t` the state corresponds to :math:`s` and the action taken by the agent is :math:`a`.

HERE IS AN IMAGE

HERE IS AN EXPLANATION

:math:`R`
*********

.. note::
   :math:`R` is the reward model. 

   :math:`R(s,a) \doteq \mathbb{E}[R_{t+1} \mid S_{t}=s, A_{t}=a]`
   
   The reward model is the function that calculates the expected value of the reward given state :math:`s` and action :math:`a` at time step :math:`t`.

HERE IS AN IMAGE

HERE IS AN EXPLANATION



:math:`\gamma`
********************

.. note::
   :math:`\gamma` (gamma) is the discound factor, where :math:`0 \leq \gamma \leq 1`.

   Gamma is used to calculate the current value of future rewards.


HERE IS AN IMAGE

HERE IS AN EXPLANATION
