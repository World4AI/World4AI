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
            name: "Definition of Machine Learning",
            link: "/blocks/deep_learning/fundamentals/machine_learning_definition",
          },
          {
            name: "Categories of Machine Learning",
            link: "/blocks/deep_learning/fundamentals/machine_learning_categories",
          },
          {
            name: "Definition of Deep Learning",
            link: "/blocks/deep_learning/fundamentals/deep_learning_definition",
          },
          {
            name: "History Of Deep Learning",
            link: "/blocks/deep_learning/fundamentals/history",
          },
          {
            name: "Deep Learning Frameworks",
            link: "/blocks/deep_learning/fundamentals/frameworks",
          },
          {
            name: "Deep Learning GPU Resources",
            link: "/blocks/deep_learning/fundamentals/gpu_resources",
          },
          {
            name: "Deep Learning Education",
            link: "/blocks/deep_learning/fundamentals/education",
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
            name: "Cross-Entropy and Negative Log Likelihood",
            link: "/blocks/deep_learning/logistic_regression/cross_entropy_negative_log_likelihood",
          },
          {
            name: "Gradient Descent",
            link: "/blocks/deep_learning/logistic_regression/gradient_descent",
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
            name: "Forward Pass",
            link: "/blocks/deep_learning/neural_network/forward_pass",
          },
          {
            name: "Backward Pass",
            link: "/blocks/deep_learning/neural_network/backward_pass",
          },
          {
            name: "Automatic Differentiation",
            link: "/blocks/deep_learning/neural_network/autodiff",
          },
          {
            name: "Geometric Interpretation",
            link: "/blocks/deep_learning/neural_network/geometric_interpretation",
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
        link: "/blocks/deep_learning/computer_vision",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/computer_vision",
          },
          {
            name: "Image Classification",
            link: "/blocks/deep_learning/computer_vision/image_classification",
          },
          {
            name: "Fundamentals of Convolutional Neural Networks",
            link: "/blocks/deep_learning/computer_vision/image_classification/convolutional_neural_networks_fundamentals",
          },
          {
            name: "CNN Architectures",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures",
          },
          {
            name: "LeNet-5",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/lenet_5",
          },
          {
            name: "AlexNet",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/alexnet",
          },
          {
            name: "VGG",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/vgg",
          },
          {
            name: "GoogLeNet",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/googlenet",
          },
          {
            name: "ResNet",
            link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/resnet",
          },
          {
            name: "Performance Tuning",
            link: "/blocks/deep_learning/computer_vision/performance_tuning",
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
            name: "Fundamentals of Recurrent Neural Networks",
            link: "/blocks/deep_learning/sequence_modelling/recurrent_neural_networks_fundamentals",
          },
          {
            name: "Types Of Recurrent Neural Networks",
            link: "/blocks/deep_learning/sequence_modelling/recurrent_neural_networks_types",
          },
          {
            name: "Biderectional Recurrent Neural Networks",
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
          {
            name: "Language Model",
            link: "/blocks/deep_learning/sequence_modelling/language_model",
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
            name: "PixelRNN",
            link: "/blocks/deep_learning/autoregressive_generative_models/pixel_rnn",
          },
          {
            name: "Gated PixelCNN",
            link: "/blocks/deep_learning/autoregressive_generative_models/gated_pixel_cnn",
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
        ],
      },
      {
        name: "Normalizing Flows",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/normalizing_flows",
          },
          {
            name: "NICE",
            link: "/blocks/deep_learning/normalizing_flows/nice",
          },
          {
            name: "RealNVP",
            link: "/blocks/deep_learning/normalizing_flows/real_nvp",
          },
          {
            name: "Glow",
            link: "/blocks/deep_learning/normalizing_flows/glow",
          },
        ],
      },
      {
        name: "Autoencoders",
        links: [
          {
            name: "Introduction",
            link: "/blocks/deep_learning/autoencoders/introduction",
          },
          {
            name: "Variational Autoencoder",
            link: "/blocks/deep_learning/autoencoders/variational_autoencoder",
          },
          {
            name: "VQ-VAE",
            link: "/blocks/deep_learning/autoencoders/vq_vae",
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
        ],
      },
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
    ],
  };
}
