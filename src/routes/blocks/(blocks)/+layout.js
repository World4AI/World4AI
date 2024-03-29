/** @type {import('./$types').LayoutServerLoad} */
export function load() {
  return {
    deep_learning: [
      {
        name: "Fundamentals",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/fundamentals/introduction",
          },
          {
            name: "Artificial Intelligence",
            link: "/blocks/deep_learning/fundamentals/artificial_intelligence",
          },
          {
            name: "Machine Learning",
            link: "/blocks/deep_learning/fundamentals/machine_learning",
          },
          {
            name: "Deep Learning",
            link: "/blocks/deep_learning/fundamentals/definition",
          },
          {
            name: "History Of Deep Learning",
            link: "/blocks/deep_learning/fundamentals/history",
          },
          {
            name: "Applications",
            link: "/blocks/deep_learning/fundamentals/applications",
          },
          {
            name: "FAQ",
            link: "/blocks/deep_learning/fundamentals/faq",
          },
          {
            name: "Mathematical Notation",
            link: "/blocks/deep_learning/fundamentals/mathematical_notation",
          },
        ],
      },
      {
        name: "Linear Regression",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/linear_regression/introduction",
          },
          {
            name: "PyTorch Tensors",
            link: "/blocks/deep_learning/linear_regression/pytorch_tensors",
          },
          {
            name: "Linear Model",
            link: "/blocks/deep_learning/linear_regression/linear_model",
          },
          {
            name: "Mean Squared Error",
            link: "/blocks/deep_learning/linear_regression/mean_squared_error",
          },
          {
            name: "Gradient Descent",
            link: "/blocks/deep_learning/linear_regression/gradient_descent",
          },
          {
            name: "Minimizing MSE",
            link: "/blocks/deep_learning/linear_regression/minimizing_mean_squared_error",
          },
          {
            name: "Linear Neuron",
            link: "/blocks/deep_learning/linear_regression/linear_neuron",
          },
        ],
      },
      {
        name: "Logistic Regression",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/logistic_regression/introduction",
          },
          {
            name: "Sigmoid and Softmax",
            link: "/blocks/deep_learning/logistic_regression/sigmoid_softmax",
          },
          {
            name: "Cross-Entropy",
            link: "/blocks/deep_learning/logistic_regression/cross_entropy",
          },
          {
            name: "Cross-Entropy vs Mean Squared Error",
            link: "/blocks/deep_learning/logistic_regression/cross_entropy_vs_mean_squared_error",
          },
          {
            name: "Minimizing Cross-Entropy",
            link: "/blocks/deep_learning/logistic_regression/minimizing_cross_entropy",
          },
          {
            name: "Sigmoid Neuron",
            link: "/blocks/deep_learning/logistic_regression/sigmoid_neuron",
          },
        ],
      },
      {
        name: "Neural Network",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/neural_network/introduction",
          },
          {
            name: "Nonlinear Problems",
            link: "/blocks/deep_learning/neural_network/nonlinear_problems",
          },
          {
            name: "Training",
            link: "/blocks/deep_learning/neural_network/training",
          },
          {
            name: "Geometric Interpretation",
            link: "/blocks/deep_learning/neural_network/geometric_interpretation",
          },
          {
            name: "Data, Modules, Optimizers, Losses",
            link: "/blocks/deep_learning/neural_network/data_modules_optimizers_losses",
          },
        ],
      },
      {
        name: "Feature Scaling",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/feature_scaling/introduction",
          },
          {
            name: "Solving MNIST",
            link: "/blocks/deep_learning/feature_scaling/solving_mnist",
          },
        ],
      },
      {
        name: "Overfitting",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/overfitting/introduction",
          },
          {
            name: "Train, Test, Validate",
            link: "/blocks/deep_learning/overfitting/train_test_validate",
          },
          {
            name: "Data Augmentation",
            link: "/blocks/deep_learning/overfitting/data_augmentation",
          },
          {
            name: "Regularization",
            link: "/blocks/deep_learning/overfitting/regularization",
          },
          {
            name: "Dropout",
            link: "/blocks/deep_learning/overfitting/dropout",
          },
          {
            name: "Early Stopping",
            link: "/blocks/deep_learning/overfitting/early_stopping",
          },
        ],
      },
      {
        name: "Vanishing and Exploding Gradients",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/vanishing_exploding_gradients/introduction",
          },
          {
            name: "Activation Functions",
            link: "/blocks/deep_learning/vanishing_exploding_gradients/activation_functions",
          },
          {
            name: "Weight Initialization",
            link: "/blocks/deep_learning/vanishing_exploding_gradients/weight_initialization",
          },
          {
            name: "Gradient Clipping",
            link: "/blocks/deep_learning/vanishing_exploding_gradients/gradient_clipping",
          },
        ],
      },
      {
        name: "Stability and Speedup",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/stability_speedup/introduction",
          },
          {
            name: "Optimizers",
            link: "/blocks/deep_learning/stability_speedup/optimizers",
          },
          {
            name: "Batch Normalization",
            link: "/blocks/deep_learning/stability_speedup/batch_normalization",
          },
          {
            name: "Skip Connections",
            link: "/blocks/deep_learning/stability_speedup/skip_connections",
          },
          {
            name: "Learning Rate Scheduling",
            link: "/blocks/deep_learning/stability_speedup/learning_rate_scheduling",
          },
        ],
      },
      {
        name: "Computer Vision",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/computer_vision/introduction",
          },
          {
            name: "Convolutional Neural Networks",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks",
          },
          {
            name: "Mixed Precision Training",
            link: "/blocks/deep_learning/computer_vision/mixed_precision_training",
          },
          {
            name: "LeNet-5",
            link: "/blocks/deep_learning/computer_vision/lenet_5",
          },
          {
            name: "AlexNet",
            link: "/blocks/deep_learning/computer_vision/alexnet",
          },
          {
            name: "VGG",
            link: "/blocks/deep_learning/computer_vision/vgg",
          },
          {
            name: "GoogLeNet",
            link: "/blocks/deep_learning/computer_vision/googlenet",
          },
          {
            name: "ResNet",
            link: "/blocks/deep_learning/computer_vision/resnet",
          },
          {
            name: "Transfer Learning",
            link: "/blocks/deep_learning/computer_vision/transfer_learning",
          },
        ],
      },
      {
        name: "Sequence Modelling",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/sequence_modelling/introduction",
          },
          {
            name: "Recurrent Neural Networks",
            link: "/blocks/deep_learning/sequence_modelling/recurrent_neural_networks",
          },
          {
            name: "Types Of RNNs",
            link: "/blocks/deep_learning/sequence_modelling/recurrent_neural_networks_types",
          },
          {
            name: "Biderectional RNNs",
            link: "/blocks/deep_learning/sequence_modelling/biderectional_recurrent_neural_networks",
          },
          {
            name: "LSTM",
            link: "/blocks/deep_learning/sequence_modelling/lstm",
          },
          {
            name: "Word Embeddings",
            link: "/blocks/deep_learning/sequence_modelling/word_embeddings",
          },
        ],
      },
      {
        name: "Attention",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/attention/introduction",
          },
          {
            name: "Bahdanau Attention",
            link: "/blocks/deep_learning/attention/bahdanau_attention",
          },
          {
            name: "Transformer",
            link: "/blocks/deep_learning/attention/transformer",
          },
          {
            name: "BERT",
            link: "/blocks/deep_learning/attention/bert",
          },
          {
            name: "GPT",
            link: "/blocks/deep_learning/attention/gpt",
          },
          {
            name: "Vision Transformer",
            link: "/blocks/deep_learning/attention/vision_transformer",
          },
        ],
      },
      {
        name: "Autoregressive Generative Models",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/autoregressive_generative_models/introduction",
          },
          {
            name: "PixelCNN",
            link: "/blocks/deep_learning/autoregressive_generative_models/pixel_cnn",
          },
          {
            name: "Gated PixelCNN",
            link: "/blocks/deep_learning/autoregressive_generative_models/gated_pixel_cnn",
          },
        ],
      },
      {
        name: "Generative Adversarial Networks",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/generative_adversarial_networks/introduction",
          },
          {
            name: "DCGAN",
            link: "/blocks/deep_learning/generative_adversarial_networks/dcgan",
          },
        ],
      },
      {
        name: "Latent Variable Models",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/latent_variable_models/introduction",
          },
          {
            name: "Autoencoders",
            link: "/blocks/deep_learning/latent_variable_models/autoencoders",
          },
          {
            name: "Variational Autoencoder",
            link: "/blocks/deep_learning/latent_variable_models/variational_autoencoder",
          },
          {
            name: "VQ-VAE",
            link: "/blocks/deep_learning/latent_variable_models/vq_vae",
          },
        ],
      },
      /*
      {
        name: "Diffusion Models",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/diffusion_models/introduction",
          },
          {
            name: "GLIDE",
            link: "/blocks/deep_learning/diffusion_models/glide",
          },
          {
            name: "Latent Diffusion",
            link: "/blocks/deep_learning/diffusion_models/latent_diffusion",
          },
        ],
      },
      */
    ],
    reinforcement_learning: [
      {
        name: "Intuition",
        links: [
          {
            name: "Introduction",
            link: "/blocks/reinforcement_learning/intuition/introduction",
          },
          {
            name: "Applications",
            link: "/blocks/reinforcement_learning/intuition/applications",
          },
          {
            name: "Agent and Environment",
            link: "/blocks/reinforcement_learning/intuition/agent_environment",
          },
          {
            name: "Definition of RL",
            link: "/blocks/reinforcement_learning/intuition/definition",
          },
          {
            name: "States, Actions, Rewards",
            link: "/blocks/reinforcement_learning/intuition/states_actions_rewards",
          },
          {
            name: "Exploration vs Exploitation",
            link: "/blocks/reinforcement_learning/intuition/exploration_exploitation_dilemma",
          },
          {
            name: "Value, Policy, Model",
            link: "/blocks/reinforcement_learning/intuition/value_policy_model",
          },
        ],
      },
      {
        name: "Markov Decision Process",
        links: [
          {
            name: "Introduction",
            link: "/blocks/reinforcement_learning/markov_decision_process/introduction",
          },
          {
            name: "Definition",
            link: "/blocks/reinforcement_learning/markov_decision_process/definition",
          },
          {
            name: "Solution",
            link: "/blocks/reinforcement_learning/markov_decision_process/solution",
          },
        ],
      },
      {
        name: "Dynamic Programming",
        links: [
          {
            name: "Introduction",
            link: "/blocks/reinforcement_learning/dynamic_programming/introduction",
          },
          {
            name: "Policy Iteration",
            link: "/blocks/reinforcement_learning/dynamic_programming/policy_iteration",
          },
          {
            name: "Value Iteration",
            link: "/blocks/reinforcement_learning/dynamic_programming/value_iteration",
          },
          {
            name: "Generalized Policy Iteration",
            link: "/blocks/reinforcement_learning/dynamic_programming/generalized_policy_iteration",
          },
        ],
      },
    ],
  };
}
