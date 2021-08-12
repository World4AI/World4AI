============
Introduction
============

So far we dealt with tabular solution methods for finite MDPs. Most interesting reinforcement learning problems do not have such nice properties. In case state or action sets are infinite or extremely large it becomes impossible to store the value function as a table. When the agent sees a continuous observation for example there is a chance that the same exact value will not be seen again. Yet the agent will need to learn how to deal with future unseen observations. We expect from the agent to find a policy that is “good”  across many different observations. The key word that we are looking for is generalization. 

In those cases it becomes necessary to create value functions that are not exact, but approximative, meaning that the value function does not return the true value of a policy but a value that is hopefully close enough. Finding the optimal policy and value function is often not possible, but the general idea is to find a policy that still performs in a desired way. 

We can look at it this way. Humans most definitely don’t have optimal policies for complex tasks they have to perform. If they did, then chess for example would be extremely boring and would not have survived for many hundred years. Still you can appreciate the complexity of the game and the extremely high level of professional players. We are going to have similar expectations for our agents. Even though we will not be able to find optimal value functions for some of the environments we are going to generate extremely impressive results.
