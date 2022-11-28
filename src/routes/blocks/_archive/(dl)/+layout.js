/** @type {import('./$types').LayoutServerLoad} */
export function load() {
  return {
    nav: [
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
            name: "Linear Regression in NumPy",
            link: "/blocks/deep_learning/linear_regression/numpy",
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
      /*
    {
      name: "Challenges and Improvements",
      link: "/blocks/deep_learning/challenges_improvements",
      links: [
        {
          name: "Feature Scaling",
          link: "/blocks/deep_learning/challenges_improvements/feature_scaling",
        },
        {
          name: "Overfitting",
          link: "/blocks/deep_learning/challenges_improvements/overfitting",
          links: [
            {
              name: "Train, Test, Validate",
              link: "/blocks/deep_learning/challenges_improvements/overfitting/train_test_validate",
            },
            {
              name: "Data Augmentation",
              link: "/blocks/deep_learning/challenges_improvements/overfitting/data_augmentation",
            },
            {
              name: "Regularization",
              link: "/blocks/deep_learning/challenges_improvements/overfitting/regularization",
            },
            {
              name: "Dropout",
              link: "/blocks/deep_learning/challenges_improvements/overfitting/dropout",
            },
            {
              name: "Early Stopping",
              link: "/blocks/deep_learning/challenges_improvements/overfitting/early_stopping",
            },
          ],
        },
        {
          name: "Vanishing and Exploding Gradients",
          link: "/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients",
          links: [
            {
              name: "Activation Functions",
              link: "/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/activation_functions",
            },
            {
              name: "Weight Initialization",
              link: "/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/weight_initialization",
            },
            {
              name: "Gradient Clipping",
              link: "/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/gradient_clipping",
            },
          ],
        },
        {
          name: "Stability and Speedup",
          link: "/blocks/deep_learning/challenges_improvements/stability_speedup",
          links: [
            {
              name: "Optimizers",
              link: "/blocks/deep_learning/challenges_improvements/stability_speedup/optimizers",
            },
            {
              name: "Batch Normalization",
              link: "/blocks/deep_learning/challenges_improvements/stability_speedup/batch_normalization",
            },
            {
              name: "Skip Connections",
              link: "/blocks/deep_learning/challenges_improvements/stability_speedup/skip_connections",
            },
            {
              name: "Learning Rate Scheduling",
              link: "/blocks/deep_learning/challenges_improvements/stability_speedup/learning_rate_scheduling",
            },
          ],
        },
      ],
    },
    {
      name: "Computer Vision",
      link: "/blocks/deep_learning/computer_vision",
      links: [
        {
          name: "Image Classification",
          link: "/blocks/deep_learning/computer_vision/image_classification",
          links: [
            {
              name: "Fundamentals of Convolutional Neural Networks",
              link: "/blocks/deep_learning/computer_vision/image_classification/convolutional_neural_networks_fundamentals",
            },
            {
              name: "Convolutional Neural Networks in PyTorch",
              link: "/blocks/deep_learning/computer_vision/image_classification/convolutional_neural_networks_pytorch",
            },
          ],
        },
        {
          name: "CNN Architectures",
          link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures",
          links: [
            {
              name: "Saving and Loading",
              link: "/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/saving_loading",
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
          ],
        },
        {
          name: "Performance Tuning",
          link: "/blocks/deep_learning/computer_vision/performance_tuning",
          links: [
            {
              name: "Mixed Precision Training",
              link: "/blocks/deep_learning/computer_vision/performance_tuning/mixed_precision_training",
            },
            {
              name: "TPU Training",
              link: "/blocks/deep_learning/computer_vision/performance_tuning/tpu_training",
            },
          ],
        },
        {
          name: "Object Detection",
          link: "/blocks/deep_learning/computer_vision/object_detection",
          links: [
            {
              name: "Intersection over Union",
              link: "/blocks/deep_learning/computer_vision/object_detection/intersection_over_union",
            },
            {
              name: "YOLO",
              link: "/blocks/deep_learning/computer_vision/object_detection/yolo",
            },
          ],
        },
        {
          name: "Image Segmentation",
          link: "/blocks/deep_learning/computer_vision/image_segmentation",
          links: [
            {
              name: "Transposed Convolutions",
              link: "/blocks/deep_learning/computer_vision/image_segmentation/transposed_convolutions",
            },
            {
              name: "U-Net",
              link: "/blocks/deep_learning/computer_vision/image_segmentation/u_net",
            },
          ],
        },
      ],
    },

    {
      name: "Sequence Modelling",
      link: "/blocks/deep_learning/sequence_modelling",
      links: [
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
        {
          name: "PyTorch Implementations",
          link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations",
          links: [
            {
              name: "Recurrent Neural Networks",
              link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations/recurrent_neural_networks",
            },
            {
              name: "Biderectional Recurrent Neural Networks",
              link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations/biderectional_recurrent_neural_networks",
            },
            {
              name: "LSTM",
              link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations/lstm",
            },
            {
              name: "Word Embeddings",
              link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations/word_embeddings",
            },
            {
              name: "Sentiment Analysis",
              link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations/sentiment_analysis",
            },
            {
              name: "Encoder-Decoder Translation",
              link: "/blocks/deep_learning/sequence_modelling/pytorch_implementations/encoder_decoder_translation",
            },
          ],
        },
      ],
    },
    {
      name: "Attention",
      link: "/blocks/deep_learning/attention",
      links: [
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
        {
          name: "Implementations",
          link: "/blocks/deep_learning/attention/implementations",
          links: [
            {
              name: "Bahdanau Attention",
              link: "/blocks/deep_learning/attention/implementations/bahdanau_attention",
            },
            {
              name: "Original Transformer",
              link: "/blocks/deep_learning/attention/implementations/transformer",
            },
            {
              name: "BERT",
              link: "/blocks/deep_learning/attention/implementations/bert",
            },
            {
              name: "GPT",
              link: "/blocks/deep_learning/attention/implementations/gpt",
            },
          ],
        },
      ],
    },
    */
    ],
  };
}
