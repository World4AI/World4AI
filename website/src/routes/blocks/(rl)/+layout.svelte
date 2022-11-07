<script>
  import Sidebar from "$lib/Sidebar.svelte";
  import Alert from "$lib/Alert.svelte";
const rl = 
[
    {
      name: 'Introduction',
      link: '/blocks/reinforcement_learning/introduction', 
      links: [
      ]
    },
    {
        name: 'Intuition',
        link: '/blocks/reinforcement_learning/intuition',
        links: [
            {
                name: 'Applications',
                link: '/blocks/reinforcement_learning/intuition/applications'
            }, {
                name: 'Agent and Environment',
                link: '/blocks/reinforcement_learning/intuition/agent_and_environment'
            },
            {
                name: 'Definition of Reinforcement Learning',
                link: '/blocks/reinforcement_learning/intuition/definition'
            },
            {
                name: 'States, Actions, Rewards',
                link: '/blocks/reinforcement_learning/intuition/states_actions_rewards'
            },
            {
                name: 'Exploration vs Exploitation',
                link: '/blocks/reinforcement_learning/intuition/exploration_vs_exploitation'
            },
            {
                name: 'Value, Policy, Model',
                link: '/blocks/reinforcement_learning/intuition/value_policy_model'
            },
            {
                name: 'Reinforcement Learning Terminology',
                link: '/blocks/reinforcement_learning/intuition/terminology'
            },
        ]
    },
    {
        name: 'Markov Decision Process',
        link: '/blocks/reinforcement_learning/markov_decision_process',
        links: [
            {
                name: 'MDP as Sequential Interaction',
                link: '/blocks/reinforcement_learning/markov_decision_process/sequential_interaction'
            },
            {
                name: 'MDP as Stochastic Process',
                link: '/blocks/reinforcement_learning/markov_decision_process/stochastic_process'
            },			
            {
                name: 'MDP as Tuple',
                link: '/blocks/reinforcement_learning/markov_decision_process/tuple'
            },			
            {
                name: 'Solution',
                link: '/blocks/reinforcement_learning/markov_decision_process/solution'
            },			
        ]
    },
    {
        name: 'Dynamic Programming',
        link: '/blocks/reinforcement_learning/dynamic_programming',
        links: [
            {
                name: 'Policy Iteration',
                link: '/blocks/reinforcement_learning/dynamic_programming/policy_iteration'
            },
            {
                name: 'Value Iteration',
                link: '/blocks/reinforcement_learning/dynamic_programming/value_iteration'
            },			
            {
                name: 'Generalized Policy Iteration',
                link: '/blocks/reinforcement_learning/dynamic_programming/generalized_policy_iteration'
            },			
        ]
    },
    {
        name: 'Exploration Exploitation',
        link: '/blocks/reinforcement_learning/exploration_exploitation_tradeoff',
        links: [
            {
                name: 'Bandits',
                link: '/blocks/reinforcement_learning/exploration_exploitation_tradeoff/bandits'
            },
            {
                name: 'Epsilon-Greedy',
                link: '/blocks/reinforcement_learning/exploration_exploitation_tradeoff/epsilon_greedy'
            },			
        ]
    },
    {
        name: 'Tabular RL',
        link: '/blocks/reinforcement_learning/tabular_reinforcement_learning',
        links: [
            {
                name: 'Monte Carlo',
                link: '/blocks/reinforcement_learning/tabular_reinforcement_learning/monte_carlo'
            },
            {
                name: 'Temporal Difference',
                link: '/blocks/reinforcement_learning/tabular_reinforcement_learning/temporal_difference'
            },			
            {
                name: 'Bias-Variance Tradeoff',
                link: '/blocks/reinforcement_learning/tabular_reinforcement_learning/bias_variance_tradeoff'
            },			
            {
                name: 'Double Q-Learning',
                link: '/blocks/reinforcement_learning/tabular_reinforcement_learning/double_q_learning'
            },			
        ]
    },
    {
        name: 'Approximative Value Function',
        link: '/blocks/reinforcement_learning/approximative_value_function',
        links: [
            {
                name: 'State and Value Representation',
                link: '/blocks/reinforcement_learning/approximative_value_function/state_value_representation'
            },
            {
                name: 'Evaluation and Improvement',
                link: '/blocks/reinforcement_learning/approximative_value_function/evaluation_improvement'
            },
            {
                name: 'Convergence and Optimality',
                link: '/blocks/reinforcement_learning/approximative_value_function/convergence_optimality'
            },			
        ]
    },
    {
        name: 'Value Based Deep Reinforcement Learning',
        link: '/blocks/reinforcement_learning/value_based_deep_reinforcement_learning',
        links: [
            {
                name: 'DQN',
                link: '/blocks/reinforcement_learning/value_based_deep_reinforcement_learning/dqn'
            },
            {
                name: 'Double DQN',
                link: '/blocks/reinforcement_learning/value_based_deep_reinforcement_learning/double_dqn'
            },
            {
                name: 'Duelling DQN',
                link: '/blocks/reinforcement_learning/value_based_deep_reinforcement_learning/duelling_dqn'
            },			
            {
                name: 'Prioritized Experience Replay',
                link: '/blocks/reinforcement_learning/value_based_deep_reinforcement_learning/prioritized_experience_replay'
            },			
        ]
    },
    {
        name: 'Policy Gradient Methods', link: '/blocks/reinforcement_learning/policy_gradient_methods',
        links: [
            {
                name: 'Policy Gradient Intuition',
                link: '/blocks/reinforcement_learning/policy_gradient_methods/policy_gradient_intuition'
            },
            {
                name: 'Policy Gradient Derivation',
                link: '/blocks/reinforcement_learning/policy_gradient_methods/policy_gradient_derivation'
            },
            {
                name: 'REINFORCE',
                link: '/blocks/reinforcement_learning/policy_gradient_methods/reinforce'
            },
            {
                name: 'Baseline',
                link: '/blocks/reinforcement_learning/policy_gradient_methods/baseline'
            },			
        ]
    },
    {
        name: 'Actor Critic Methods',
        link: '/blocks/reinforcement_learning/actor_critic_methods',
        links: [
            {
                name: 'A3C and A2C',
                link: '/blocks/reinforcement_learning/actor_critic_methods/a3c_a2c'
            },
            {
                name: 'Generalized Advantage Estimation (GAE)',
                link: '/blocks/reinforcement_learning/actor_critic_methods/generalized_advantage_estimation'
            },
        ]
    },
    {
        name: 'Trust Region Methods',
        link: '/blocks/reinforcement_learning/trust_region_methods',
        links: [
            {
                name: 'Trust Region Policy Optimization (TRPO)',
                link: '/blocks/reinforcement_learning/trust_region_methods/trust_region_policy_optimization'
            },
            {
                name: 'Proximal Policy Optimization (PPO)',
                link: '/blocks/reinforcement_learning/trust_region_methods/proximal_policy_optimization'
            },
        ]
    },
]
</script>

<Sidebar root={rl} />
<main>
  <article>
    <Alert type="danger">The reinforcement learning block is outdated and needs a lot of refactoring. </Alert>
    <slot />
  </article>
</main>
