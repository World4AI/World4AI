<script>
  import Sidebar from "$lib/Sidebar.svelte";


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
        name: 'PyTorch Basics',
        link: '/blocks/deep_learning/pytorch_basics',
        links: [
          {
            name: 'Tensors',
            link:  '/blocks/deep_learning/pytorch_basics/tensors'
          },
          {
            name: 'Autograd',
            link:  '/blocks/deep_learning/pytorch_basics/autograd'
          },
          {
            name: 'Data',
            link:  '/blocks/deep_learning/pytorch_basics/data'
          },
          {
            name: 'Training Loop',
            link:  '/blocks/deep_learning/pytorch_basics/training_loop'
          },
          {
            name: 'Modules, Optimizers, Losses',
            link:  '/blocks/deep_learning/pytorch_basics/modules_optimizers_losses'
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
            links: [
              {
                name: 'Feature Scaling in PyTorch',
                link:  '/blocks/deep_learning/challenges_improvements/feature_scaling/pytorch'
              },
            ]
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
              {
                name: 'Dealing with Overfitting in PyTorch',
                link:  '/blocks/deep_learning/challenges_improvements/overfitting/pytorch'
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
              {
                name: 'Dealing with Problematic Gradients in PyTorch',
                link:  '/blocks/deep_learning/challenges_improvements/vanishing_exploding_gradients/pytorch'
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
                link:  '/blocks/deep_learning/challenges_improvements/stability_speedup/skip_connections'
              },
              {
                name: 'Learning Rate Scheduling',
                link:  '/blocks/deep_learning/challenges_improvements/stability_speedup/learning_rate_scheduling'
              },
              {
                name: 'Stability and Speedup in PyTorch',
                link:  '/blocks/deep_learning/challenges_improvements/stability_speedup/pytorch'
              },
            ]
          },
        ]
    },
  {
    name: 'Computer Vision',
    link: '/blocks/deep_learning/computer_vision',
    links: [
      {
        name: 'Image Classification',
        link: '/blocks/deep_learning/computer_vision/image_classification',
        links: [
          {
            name: 'Fundamentals of Convolutional Neural Networks',
            link: '/blocks/deep_learning/computer_vision/image_classification/convolutional_neural_networks_fundamentals',
          },
          {
            name: 'Convolutional Neural Networks in PyTorch',
            link: '/blocks/deep_learning/computer_vision/image_classification/convolutional_neural_networks_pytorch',
          },
        ]
      },
      {
        name: 'CNN Architectures',
        link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures',
        links: [
          {
            name: 'Saving and Loading',
            link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/saving_loading',
          },
          {
            name: 'LeNet-5',
            link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/lenet_5',
          },
          {
            name: 'AlexNet',
            link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/alexnet',
          },
          {
            name: 'VGG',
            link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/vgg',
          },
          {
            name: 'GoogLeNet',
            link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/googlenet',
          },
          {
            name: 'ResNet',
            link: '/blocks/deep_learning/computer_vision/convolutional_neural_networks_architectures/resnet',
          },
        ]
      },
      {
        name: 'Performance Tuning',
        link: '/blocks/deep_learning/computer_vision/performance_tuning',
        links: [
          {
            name: 'Mixed Precision Training',
            link: '/blocks/deep_learning/computer_vision/performance_tuning/mixed_precision_training',
          },
          {
            name: 'TPU Training',
            link: '/blocks/deep_learning/computer_vision/performance_tuning/tpu_training',
          },
        ]
      },
      {
        name: 'Object Detection',
        link: '/blocks/deep_learning/computer_vision/object_detection',
        links: [
          {
            name: 'Intersection over Union',
            link: '/blocks/deep_learning/computer_vision/object_detection/intersection_over_union',
          },
          {
            name: 'YOLO',
            link: '/blocks/deep_learning/computer_vision/object_detection/yolo',
          },
        ]
      },
      {
        name: 'Image Segmentation',
        link: '/blocks/deep_learning/computer_vision/image_segmentation',
        links: [
          {
            name: 'Transposed Convolutions',
            link: '/blocks/deep_learning/computer_vision/image_segmentation/transposed_convolutions',
          },
          {
            name: 'U-Net',
            link: '/blocks/deep_learning/computer_vision/image_segmentation/u_net',
          },
        ]
      },
    ]
 },

    {
        name: 'Sequence Modelling',
        link: '/blocks/deep_learning/sequence_modelling',
        links: [
          {
            name: 'Fundamentals of Recurrent Neural Networks',
            link: '/blocks/deep_learning/sequence_modelling/recurrent_neural_networks_fundamentals',
          },
          {
            name: 'Types Of Recurrent Neural Networks',
            link: '/blocks/deep_learning/sequence_modelling/recurrent_neural_networks_types',
          },
          {
            name: 'Biderectional Recurrent Neural Networks',
            link: '/blocks/deep_learning/sequence_modelling/biderectional_recurrent_neural_networks',
          },
          {
            name: 'LSTM',
            link: '/blocks/deep_learning/sequence_modelling/lstm',
          },
          {
            name: 'Word Embeddings',
            link: '/blocks/deep_learning/sequence_modelling/word_embeddings',
          },
          {
            name: 'Language Model',
            link: '/blocks/deep_learning/sequence_modelling/language_model',
          },
          {
            name: 'PyTorch Implementations',
            link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations',
            links: [
              {
                name: 'Recurrent Neural Networks',
                link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations/recurrent_neural_networks',
              },
              {
                name: 'Biderectional Recurrent Neural Networks',
                link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations/biderectional_recurrent_neural_networks',
              },
              {
                name: 'LSTM',
                link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations/lstm',
              },
              {
                name: 'Word Embeddings',
                link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations/word_embeddings',
              },
              {
                name: 'Sentiment Analysis',
                link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations/sentiment_analysis',
              },
              {
                name: 'Encoder-Decoder Translation',
                link: '/blocks/deep_learning/sequence_modelling/pytorch_implementations/encoder_decoder_translation',
              }
            ]
          },
        ]
    },
    {
        name: 'Attention',
        link: '/blocks/deep_learning/attention',
        links: [ 
          {
            name: 'Bahdanau Attention',
            link: '/blocks/deep_learning/attention/bahdanau_attention',
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
          {
            name: 'Vision Transformer',
            link: '/blocks/deep_learning/attention/vision_transformer',
          },
          {
            name: 'Implementations',
            link: '/blocks/deep_learning/attention/implementations',
            links: [
              {
                name: 'Bahdanau Attention',
                link: '/blocks/deep_learning/attention/implementations/bahdanau_attention',
              },
              {
                name: 'Original Transformer',
                link: '/blocks/deep_learning/attention/implementations/transformer',
              },
              {
                name: 'BERT',
                link: '/blocks/deep_learning/attention/implementations/bert',
              },
              {
                name: 'GPT',
                link: '/blocks/deep_learning/attention/implementations/gpt',
              }
            ]
          },
        ]
    },
]

</script>

<Sidebar root={dl} />
<main>
  <article>
    <slot />
  </article>
</main>
