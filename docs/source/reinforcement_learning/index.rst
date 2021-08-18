======================
Reinforcement Learning
======================

In this introduction I will go over the possible applications, the key concepts and the components of reinforcement learning. The focus is to build up intuition before diving deep into mathematics.


.. toctree::
   :maxdepth: 2
   :caption: Introduction

   introduction/applications
   introduction/agent_env
   introduction/properties
   introduction/interaction
   introduction/exploration_vs_exploitation
   introduction/components


.. toctree::
   :maxdepth: 2
   :caption: Markov Decision Process

   markov_decision_process/introduction
   markov_decision_process/definition_markov_decision_process
   markov_decision_process/to_solve_mdp

.. toctree::
   :maxdepth: 2
   :caption: Dynamic Programming

   dynamic_programming/introduction
   dynamic_programming/policy_iteration
   dynamic_programming/value_iteration
   dynamic_programming/generalized_policy_iteration

.. toctree::
   :maxdepth: 2
   :caption: Exploration Exploitation Tradeoff

   exploration_exploitation_tradeoff/introduction
   exploration_exploitation_tradeoff/bandits
   exploration_exploitation_tradeoff/epsilon_greedy

.. toctree::
   :maxdepth: 2
   :caption: Tabular Reinforcement Learning

   tabular_reinforcement_learning/introduction
   tabular_reinforcement_learning/monte_carlo_methods
   tabular_reinforcement_learning/td_learning

.. toctree::
   :maxdepth: 2
   :caption: Approximative Reinforcement Learning

   approximative_reinforcement_learning/introduction
   approximative_reinforcement_learning/approximation_with_oracle

.. toctree::
   :maxdepth: 2
   :caption: Modern Value-Based Approximation

   modern_value_based_approximation/introduction
   modern_value_based_approximation/nfq
   modern_value_based_approximation/dqn
   modern_value_based_approximation/double_dqn
   modern_value_based_approximation/duelling_dqn

.. toctree::
   :maxdepth: 2
   :caption: Reinforcement Learning Libraries

   rl_libraries/motivation
   rl_libraries/openai_gym