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
            name: 'Intuition',
            link: '/blocks/deep_learning/introduction/intuition'
          },
          {
            name: 'Definition',
            link: '/blocks/deep_learning/introduction/definition'
          },
          {
            name: 'History',
            link: '/blocks/deep_learning/introduction/history'
          },
          {
            name: 'Applications',
            link: '/blocks/deep_learning/introduction/applications'
          },
          {
            name: 'Frameworks',
            link: '/blocks/deep_learning/introduction/frameworks'
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
        name: 'Loss Functions',
        link: '/blocks/deep_learning/loss_functions',
        links: [
        ]
    },
    {
        name: 'Backpropagation',
        link: '/blocks/deep_learning/backpropagation',
        links: [
          {
            name: 'Autodiff',
            link: '/blocks/deep_learning/backpropagation/autodiff',
          }
        ]
    },
    {
        name: 'Vanishing and Exploding Gradients',
        link: '/blocks/deep_learning/vanishing_exploding_gradients',
        links: [
        ]
    },
    {
        name: 'Activations',
        link: '/blocks/deep_learning/activations',
        links: [
        ]
    },
    {
        name: 'Optimizers',
        link: '/blocks/deep_learning/optimizers',
        links: [
          {
            name: 'SGD',
            link: '/blocks/deep_learning/optimizers/sgd',
          },
          {
            name: 'Momentum',
            link: '/blocks/deep_learning/optimizers/momentum',
          },
          {
            name: 'RMSProp',
            link: '/blocks/deep_learning/optimizers/rmsprop',
          },
          {
            name: 'Adam',
            link: '/blocks/deep_learning/optimizers/adam',
          },
        ]
    },
    {
        name: 'Regularization',
        link: '/blocks/deep_learning/regularization',
        links: [
        ]
    },
    {
        name: 'Initialization and Normalization',
        link: '/blocks/deep_learning/initialization_normalization',
        links: [
          {
            name: 'Input Normalization',
            link: '/blocks/deep_learning/initialization_normalization/input_normalization',
          },
          {
            name: 'Weight Initialization',
            link: '/blocks/deep_learning/initialization_normalization/weight_initialization',
          },
          {
            name: 'Batch Normalization',
            link: '/blocks/deep_learning/initialization_normalization/batch_normalization',
          },
          {
            name: 'Layer Normalization',
            link: '/blocks/deep_learning/initialization_normalization/layer_normalization',
          },
        ]
    },
    {
        name: 'Convolutional Neural Networks',
        link: '/blocks/deep_learning/convolutional_neural_networks',
        links: [
        ]
    },
    {
        name: 'Convolutional Neural Networks Architectures',
        link: '/blocks/deep_learning/convolutional_neural_networks_architectures',
        links: [
          {
            name: 'LeNet',
            link: '/blocks/deep_learning/convolutional_neural_networks_architectures/lenet',
          },
          {
            name: 'AlexNet',
            link: '/blocks/deep_learning/convolutional_neural_networks_architectures/alexnet',
          },
          {
            name: 'VGG',
            link: '/blocks/deep_learning/convolutional_neural_networks_architectures/vgg',
          },
          {
            name: 'GoogleNet',
            link: '/blocks/deep_learning/convolutional_neural_networks_architectures/googlenet',
          },
          {
            name: 'ResNet',
            link: '/blocks/deep_learning/convolutional_neural_networks_architectures/resnet',
          },
          {
            name: 'EfficientNet',
            link: '/blocks/deep_learning/convolutional_neural_networks_architectures/efficientnet',
          },
        ]
    },
    {
        name: 'Convolutional Neural Networks Applications',
        link: '/blocks/deep_learning/convolutional_neural_networks_applications',
        links: [
          {
            name: 'Object Classification',
            link: '/blocks/deep_learning/convolutional_neural_networks_applications/object_classification',
          },
          {
            name: 'Object Detection',
            link: '/blocks/deep_learning/convolutional_neural_networks_applications/object_detection',
          },
          {
            name: 'Segmentation',
            link: '/blocks/deep_learning/convolutional_neural_networks_applications/segmentation',
          },
          {
            name: 'Neural Style Transfer',
            link: '/blocks/deep_learning/convolutional_neural_networks_applications/neural_style_transfer',
          },
        ]
    },
    {
        name: 'Word Embeddings',
        link: '/blocks/deep_learning/word_embeddings',
        links: [
        ]
    },
    {
        name: 'Recurrent Neural Networks',
        link: '/blocks/deep_learning/recurrent_neural_networks',
        links: [
          {
            name: 'Long Short-Term Memory',
            link: '/blocks/deep_learning/recurrent_neural_networks/long_short_term_memory',
          },
          {
            name: 'Gated Recurrent Units',
            link: '/blocks/deep_learning/recurrent_neural_networks/gated_recurrent_units',
          },
        ]
    },
    {
        name: 'Generative Models',
        link: '/blocks/deep_learning/generative_models',
        links: [
          {
            name: 'Autoencoders',
            link: '/blocks/deep_learning/generative_models/autoencoders',
          },
          {
            name: 'Variational Autoencoders',
            link: '/blocks/deep_learning/generative_models/variational_autoencoders',
          },
          {
            name: 'Generative Adversarial Networks',
            link: '/blocks/deep_learning/generative_models/generative_adversarial_networks',
          },
          {
            name: 'Diffusion Models',
            link: '/blocks/deep_learning/generative_models/diffusion_models',
          },
        ]
    },
    {
        name: 'Attention',
        link: '/blocks/deep_learning/attention',
        links: [ 
          {
            name: 'RNNs with Attention',
            link: '/blocks/deep_learning/attention/rnn_attention',
          },
          {
            name: 'Transformer',
            link: '/blocks/deep_learning/attention/transformer',
          },
          {
            name: 'BERT',
            link: '/blocks/deep_learning/attention/bert',
          },
          {
            name: 'GPT',
            link: '/blocks/deep_learning/attention/gpt',
          },
        ]
    },
    {
        name: 'Graph Neural Networks',
        link: '/blocks/deep_learning/graph_neural_networks',
        links: [
        ]
    },
]
export {rl, dl, programming, mathematics}
