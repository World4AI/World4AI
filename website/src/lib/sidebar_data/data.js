const rl = [
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
            },
            {
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
        name: 'Policy Gradient Methods',
        link: '/blocks/reinforcement_learning/policy_gradient_methods',
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
];

const programming = [
    {
        name: 'Introduction',
        link: '/blocks/programming/introduction',
        links: [
            {
                name: 'Python',
                link: '/blocks/programming/introduction/python'
            },
            {
                name: 'Setup',
                link: '/blocks/programming/introduction/setup'
            },
        ]
    },
    {
        name: 'Fundamentals',
        link: '/blocks/programming/fundamentals',
        links: [
            {
                name: 'Getting Help',
                link: '/blocks/programming/fundamentals/getting_help'
            },
            {
                name: 'Hello World',
                link: '/blocks/programming/fundamentals/hello_world'
            },
            {
                name: 'Commenting',
                link: '/blocks/programming/fundamentals/commenting'
            },
            {
                name: 'Objects',
                link: '/blocks/programming/fundamentals/objects'
            },
            {
                name: 'Variables',
                link: '/blocks/programming/fundamentals/variables'
            },
            {
                name: 'Data Types',
                link: '/blocks/programming/fundamentals/data_types'
            },
            {
                name: 'Garbage Collector',
                link: '/blocks/programming/fundamentals/garbage_collector'
            },
            {
                name: 'Operators',
                link: '/blocks/programming/fundamentals/operators'
            },
       ]
    },
    {
        name: 'Data Types',
        link: '/blocks/programming/data_types',
        links: [
            {
                name: 'Numeric',
                link: '/blocks/programming/data_types/numeric'
            },
            {
                name: 'String',
                link: '/blocks/programming/data_types/string'
            },
            {
                name: 'Boolean',
                link: '/blocks/programming/data_types/boolean'
            },
            {
                name: 'Tuple',
                link: '/blocks/programming/data_types/tuple'
            },
            {
                name: 'List',
                link: '/blocks/programming/data_types/list'
            },
            {
                name: 'Dictionary',
                link: '/blocks/programming/data_types/dictionary'
            },
            {
                name: 'Casting',
                link: '/blocks/programming/data_types/casting'
            },
            {
                name: 'Dynamic and Strong Typing',
                link: '/blocks/programming/data_types/dynamic_strong_typing'
            },
        ]
    },
    {
        name: 'Functions',
        link: '/blocks/programming/functions',
        links: [
            {
                name: 'Basic Usage',
                link: '/blocks/programming/functions/basic_usage'
            },
            {
                name: 'Inputs and Outputs',
                link: '/blocks/programming/functions/inputs_outputs'
            },
        ]
    },
    {
        name: 'Control Flow',
        link: '/blocks/programming/control_flow',
        links: [
            {
                name: 'Conditionals',
                link: '/blocks/programming/control_flow/conditionals'
            },
            {
                name: 'Loops',
                link: '/blocks/programming/control_flow/loops'
            },
        ]
    },
    {
        name: 'Object Oriented Programming (OOP)',
        link: '/blocks/programming/object_oriented_programming',
        links: [
            {
                name: 'Classes and Objects',
                link: '/blocks/programming/object_oriented_programming/classes_objects'
            },
            {
                name: 'The Four Pillars of OOP',
                link: '/blocks/programming/object_oriented_programming/four_pillars'
            },
        ]
    },
];

const mathematics = [
    {
        name: 'Introduction',
        link: '/blocks/mathematics/introduction',
        links: [
        ]
    },
    {
        name: 'Linear Algebra',
        link: '/blocks/mathematics/linear_algebra',
        links: [
        ]
    },
    {
        name: 'Calculus',
        link: '/blocks/mathematics/calculus',
        links: [
        ]
    },
    {
        name: 'Probability Theory',
        link: '/blocks/mathematics/probability_theory',
        links: [
        ]
    },
    {
        name: 'Information Theory',
        link: '/blocks/mathematics/information_theory',
        links: [
          {
            name: 'Information',
            link: '/blocks/mathematics/information_theory/information'
          },
          {
            name: 'Entropy',
            link: '/blocks/mathematics/information_theory/entropy'
          },
          {
            name: 'Cross Entropy',
            link: '/blocks/mathematics/information_theory/cross_entropy'
          },
          {
            name: 'KL Divergence',
            link: '/blocks/mathematics/information_theory/kl_divergence'
          },
        ]
    },
];

const dl = [
    {
        name: 'Introduction',
        link: '/blocks/deep_learning/introduction',
        links: [
          {
            name: 'History',
            link: '/blocks/deep_learning/introduction/history'
          }
        ]
    },
    {
        name: 'Linear Regression',
        link: '/blocks/deep_learning/linear_regression',
        links: [
        ]
    },
    {
        name: 'Perceptron',
        link: '/blocks/deep_learning/perceptron',
        links: [
        ]
    },
    {
        name: 'Logistic Regression',
        link: '/blocks/deep_learning/logistic_regression',
        links: [
        ]
    },
    {
        name: 'Neural Network',
        link: '/blocks/deep_learning/neural_network',
        links: [
        ]
    },
    {
        name: 'Backpropagation',
        link: '/blocks/deep_learning/backpropagation',
        links: [
        ]
    },
    {
        name: 'Convolutional Neural Networks',
        link: '/blocks/deep_learning/convolutional_neural_networks',
        links: [
        ]
    },
    {
        name: 'Recurrent Neural Networks',
        link: '/blocks/deep_learning/recurrent_neural_networks',
        links: [
        ]
    },
]
export {rl, dl, programming, mathematics}
