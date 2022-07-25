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
];

const dl = [
    {
        name: 'Introduction',
        link: '/blocks/deep_learning/introduction',
        links: [
          {
            name: 'Definition of Machine Learning',
            link: '/blocks/deep_learning/introduction/machine_learning_definition'
          },
          {
            name: 'Categories of Machine Learning',
            link: '/blocks/deep_learning/introduction/machine_learning_categories'
          },
          {
            name: 'Definition of Deep Learning',
            link: '/blocks/deep_learning/introduction/deep_learning_definition'
          },
          {
            name: 'History Of Deep Learning',
            link: '/blocks/deep_learning/introduction/history'
          },
          {
            name: 'Deep Learning Frameworks',
            link: '/blocks/deep_learning/introduction/frameworks'
          },
          {
            name: 'Deep Learning GPU Resources',
            link: '/blocks/deep_learning/introduction/gpu_resources'
          },
          {
            name: 'Deep Learning Education',
            link: '/blocks/deep_learning/introduction/education'
          },
          {
            name: 'Mathematical Notation',
            link: '/blocks/deep_learning/introduction/mathematical_notation'
          },
        ]
    },
    {
        name: 'Linear Regression',
        link: '/blocks/deep_learning/linear_regression',
        links: [
          {
            name: 'Linear Model',
            link:'/blocks/deep_learning/linear_regression/linear_model', 
          },
          {
            name: 'Mean Squared Error',
            link:'/blocks/deep_learning/linear_regression/mean_squared_error', 
          },
          {
            name: 'Gradient Descent',
            link:'/blocks/deep_learning/linear_regression/gradient_descent', 
          },
          {
            name: 'Linear Regression in NumPy',
            link:'/blocks/deep_learning/linear_regression/numpy', 
          },
          {
            name: 'Linear Neuron',
            link:'/blocks/deep_learning/linear_regression/linear_neuron', 
          },
        ]
    },
    {
        name: 'Logistic Regression',
        link: '/blocks/deep_learning/logistic_regression',
        links: [
          {
              name: 'Sigmoid and Softmax',
              link: '/blocks/deep_learning/logistic_regression/sigmoid_softmax',
          },
          {
              name: 'Cross-Entropy and Negative Log Likelihood',
              link: '/blocks/deep_learning/logistic_regression/cross_entropy_negative_log_likelihood',
          },
          {
              name: 'Gradient Descent',
              link: '/blocks/deep_learning/logistic_regression/gradient_descent',
          },
          {
              name: 'Logistic Regression in NumPy',
              link: '/blocks/deep_learning/logistic_regression/numpy',
          },
          {
              name: 'Sigmoid Neuron',
              link: '/blocks/deep_learning/logistic_regression/sigmoid_neuron',
          },
        ]
    },
    {
        name: 'Neural Network',
        link: '/blocks/deep_learning/neural_network',
        links: [
          {
            name: 'Nonlinear Problems',
            link:  '/blocks/deep_learning/neural_network/nonlinear_problems'
          },
          {
            name: 'Forward Pass',
            link:  '/blocks/deep_learning/neural_network/forward_pass'
          },
          {
            name: 'Backward Pass',
            link:  '/blocks/deep_learning/neural_network/backward_pass'
          },
          {
            name: 'Backpropagation in NumPy',
            link:  '/blocks/deep_learning/neural_network/numpy'
          },
          {
            name: 'Automatic Differentiation',
            link:  '/blocks/deep_learning/neural_network/autodiff'
          },
          {
            name: 'Geometric Interpretation',
            link:  '/blocks/deep_learning/neural_network/geometric_interpretation'
          },
        ]
    },
    {
        name: 'Challenges and Improvements',
        link: '/blocks/deep_learning/challenges_improvements',
        links: [
          {
            name: 'Feature Scaling',
            link:  '/blocks/deep_learning/challenges_improvements/feature_scaling',
          },
          {
            name: 'Overfitting',
            link:  '/blocks/deep_learning/challenges_improvements/overfitting',
            links: [
              {
                name: 'Train, Test, Validate',
                link:  '/blocks/deep_learning/challenges_improvements/overfitting/train_test_validate'
              },
              {
                name: 'Data Augmentation',
                link:  '/blocks/deep_learning/challenges_improvements/overfitting/data_augmentation'
              },
              {
                name: 'Regularization',
                link:  '/blocks/deep_learning/challenges_improvements/overfitting/regularization'
              },
              {
                name: 'Dropout',
                link:  '/blocks/deep_learning/challenges_improvements/overfitting/dropout'
              },
              {
                name: 'Early Stopping',
                link:  '/blocks/deep_learning/challenges_improvements/overfitting/early_stopping'
              },
            ]
          },
          {
            name: 'Vanishing and Exploding Gradients',
            link:  '/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients',
            links: [
              {
                name: 'Activation Functions',
                link:  '/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/activation_functions'
              },
              {
                name: 'Weight Initialization',
                link:  '/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/weight_initialization'
              },
              {
                name: 'Gradient Clipping',
                link:  '/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/gradient_clipping'
              },
            ]
          },
          {
            name: 'Stability and Speedup',
            link:  '/blocks/deep_learning/challenges_improvements/stability_speedup',
            links: [
              {
                name: 'Optimizers',
                link:  '/blocks/deep_learning/challenges_improvements/stability_speedup/optimizers'
              },
              {
                name: 'Batch Normalization',
                link:  '/blocks/deep_learning/challenges_improvements/stability_speedup/batch_normalization'
              },
              {
                name: 'Skip Connections',
                link:  '/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/skip_connections'
              },
              {
                name: 'Learning Rate Scheduling',
                link:  '/blocks/deep_learning/challenges_improvements/stability_speedup/learning_rate_scheduling'
              },
            ]
          },
          {
             name: 'Transfer Learning',
             link:  '/blocks/deep_learning/challenges_improvements/transfer_learning'
          },
        ]
    },
  /*
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
    */
]
export {dl, rl}
