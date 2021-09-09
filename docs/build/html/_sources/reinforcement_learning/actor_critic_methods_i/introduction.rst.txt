============
Introduction
============

So far we have seen value based methods like DQN, which estimate state or action value function and policy based methods like REINFORCE, which estimate the policy directly. It turns out that combining both types of methods can result in so-called actor-critic methods. The actor is the decision maker and the policy of the agent. The critic is the value function that estimates how good or bad the decisions are that the critic makes. Both are usually implemented as neural networks and trained simultaneously. The actor-critic methods can have significant improvements over pure value or policy based methods and in many cases constitute state of the art methods that are available to us at this point in time.
