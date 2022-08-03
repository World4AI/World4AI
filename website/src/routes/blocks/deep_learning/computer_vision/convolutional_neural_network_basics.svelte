<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Flattening from "./_convolution/Flattening.svelte";
  import Convolution from "./_convolution/Convolution.svelte";
  import Latex from "$lib/Latex.svelte";

  const imageOrig = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
  ];

  const imageShifted = [
    [0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ];

  export let layers = [
    {
      title: "Input",
      nodes: [
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
      ],
    },
    {
      title: "Hidden Layer",
      nodes: [
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
      ],
    },
    {
      title: "",
      nodes: [{ value: "", fill: "none" }],
    },
  ];
</script>

<h1>The Fundamentals of Convolutional Neural Networks</h1>
<div class="separator" />
<Container>
  <p>
    Let's start this section with the following question: "Why is a fully
    connected neural network not the ideal tool for computer vision and how
    would we design a neural network that is a much better fit for images and
    video?".
  </p>
  <p>
    When we deal with images in PyTorch we are dealing with images in <Highlight
      >NCWH</Highlight
    >
    format.
  </p>
  <ul>
    <li>N is the number of images in a batch</li>
    <li>C is the number of channels (1 for greyscale, 3 for color)</li>
    <li>W is the width of the image</li>
    <li>H is the height of the image</li>
  </ul>
  <p>
    The tensor with dimensions (32, 3, 28, 28) for example tells us that we have
    a batch of 32 images with 3 channels. Each color channel has a height and a
    width of 28 pixels. In the last section when we dealt with MNIST images we
    had images of shape (32, 1, 28, 28). But the neural networks that we used so
    far can not process those images directly. The input needs to be a vector.
    So we flattened the image into the shape (32, 784), thereby losing all
    spatial information.
  </p>
  <p>
    To show that spatial information is extremely useful, let's assume for a
    second, that we encounter the digit 2 from the below image. If you look at
    the digit directly, you will have no problem recognizing the number. But
    when you interract with the example and flatten the image (as we did with
    MNIST), the task gets a lot harder. Yet that is exactly the problem our
    fully connected neural network has to solve.
  </p>
  <Flattening image={imageOrig} />
  <p>
    The loss of all spatial information makes our model also quite sensitive to
    different types of transformations: like translation, rotation, scaling,
    color and lightning. The image below is shifted sligtly to the rigth and to
    the top. When you compare the two flattened images you will notices, that
    there is hardly any overlap in pixel values, even thought we are dealing
    with an almost identical image.
  </p>
  <Flattening image={imageShifted} />
  <p>
    Even if we didn't lose any spatial information, the combination of a fully
    connected neural network and images is problematic. The neural network below
    depicts a greyscale image of size 4x4 pixels. You can hardly argue that that
    is an image at all, yet the 16 inputs and the ten neurons in the hidden
    layer require 160 weights and 10 biases.
  </p>
  <NeuralNetwork {layers} height={250} verticalGap={5} rectSize={9} />
  <p>
    Real-life images are vastly larger than that and we require a neural network
    with hundreds or thousands of neurons and several hidden layers to solve
    even a simple task. Even for an image of size 100x100 and 100 neuron we are
    dealing with 1,000,000 weights. At a certain point training fully connected
    neural network is going to become extremely inefficient.
  </p>
  <p>
    Given those problems, we need a new neural network architecture. An
    architecture that is able to deal with image and video without destroying
    spatial information and requires much fewer learnable parameters at the same
    time. The neural network that would alleviate our problems is called a <Highlight
      >convolution neural network</Highlight
    >, often abbreviated as CNN.
  </p>
  <div class="separator" />

  <h2>Convolutional Layer</h2>
  <p>
    Let's take it one step at a time and think about how we could design such a
    neural network. We start with a basic assumption: "pixels in a small region
    of an image are highly correlated". If you look at an image that contains
    the sky and you pick any pixel of the sky, then with a very high probability
    the connecting pixels that surround that location are going to be part of
    the sky as well. In essence that is what we mean when we talk about spatial
    information. Look at your surroundings. Basically all objects exhibit forms
    and colors that are extremely similar at some local patch. The spots where
    the similarity ends is where you start to notice distinct objects or
    patterns within objects.
  </p>
  <p>
    In order to somehow leverage the spatial information that is contained in a
    local patch we could require a neural network that limits the receptive
    field of each neuron. A neuron gets assigned a small patch in the image and
    can only attend to that small patch. Below for example the first neuron in
    the first hidden layer would focus only on the top left corner.
  </p>
  <Convolution maxWidth={200} kernel={2} />
  <p>
    In a fully connected neural network a neuron had to be connected to all
    input pixels (therefore the name fully connected). If we limit the number of
    pixels to a local patch of 2x2, that reduces the number of weights for a
    single neuron from 28*28 (MNIST dimensions) to just four. This is called <Highlight
      >sparse connectivity</Highlight
    >.
  </p>
  <p>
    Each neuron is calculated using a different patch of pixels. You can imagine
    that those calculations are conducted by using a sliding window, therefore
    changing the receptive field of each neuron in the hidden layer. The neurons
    themselves are placed in a way that keeps the structure of the image. The
    neuron that has the upper left corner in its receptive field is located in
    the upper left corner of the hidden layer. The neuron that attends to the
    patch that is to the right of the upper left corner, is put to the right of
    the before mentioned neuron. When the receptive field moves a row below, the
    neurons that attend to that receptive field also move below. This results in
    a two dimensional image. You can start the interactive example and observe
    how the receptive field moves and how the neurons are placed in a 2D grid.
    Notice also that the output image shrinks. This is expected, because a 2x2
    patch is required to construct a single neuron.
  </p>
  <Convolution maxWidth={500} kernel={2} showOutput={true} />
  <p>
    You have a lot of control over the behaviour of the receptive field. You can
    for example control the size of the receptive field. Above we used the
    window of size 2x2, but 3x3 is also a common size for the receptive field.
  </p>
  <Convolution maxWidth={500} kernel={3} stride={1} showOutput={true} />
  <p>
    The <Highlight>stride</Highlight> is also a hyperparameter you wil be interested
    in. The stride controls the number of steps the window is moved. Above the window
    was moved 1 step to right and 1 step below, which corresponds to a stride of
    1. In the example below we use a stride of 2. A larger stride obviously makes
    the output image smaller.
  </p>
  <Convolution maxWidth={450} kernel={2} stride={2} showOutput={true} />
  <p>
    As you have probability noticed, the output image is always smaller than the
    input image. If you want to keep the dimensionality between the input and
    ouput images consistent, you can pad the input image. Basically that means
    that you add artificial pixels by surrounding the input image with zeros
    (yellow boxes in the image below).
  </p>
  <Convolution
    maxWidth={650}
    imageWidth={5}
    imageHeight={5}
    kernel={3}
    stride={1}
    padding={1}
    showOutput={true}
  />
  <p>
    When it comes to the calculation of the neuron values, we are dealing with
    an almost identical procedure that we used in the previous chapters. Let's
    asume we want to calculate the activation value for the patch in the upper
    left corner.
  </p>
  <Convolution maxWidth={200} kernel={2} />
  <p>
    The patch
    <Latex
      >{String.raw`
  \begin{bmatrix}
     x_{11} & x_{12} \\
     x_{21} & x_{22}
  \end{bmatrix}
  `}</Latex
    >
    is <Latex>{String.raw`2\times2`}</Latex>, therefore we need exactly 4
    weights.
    <Latex
      >{String.raw`
  \begin{bmatrix}
     w_{11} & w_{12} \\
     w_{21} & w_{22}
  \end{bmatrix}
  `}</Latex
    >
  </p>
  <p>
    This collection of weights that is applied to a limited receptive field is
    called a <Highlight>filter</Highlight> or a <Highlight>kernel</Highlight>.
  </p>
  <p>
    Similar to a fully connected neural network we calcualate a weighted sum,
    add a bias and apply a non-linear activation function to get the value of a
    neuron in the next layer.
  </p>
  <Latex
    >{String.raw`
  \begin{aligned}
  z &= x_{11}w_{11} + x_{12}w_{12} + x_{21}w_{21} + x_{22}w_{22} + b \\
  a &= \sigma(z)
  \end{aligned}
  `}</Latex
  >
  <p>
    What is unique about convolutional neural networks is the <Highlight
      >weight sharing</Highlight
    > among all neurons. When we slide the window of the receptive field, we do not
    replace the weights and biases, but always keep the same identical filter
    <Latex
      >{String.raw`
  \begin{bmatrix}
     w_{1} & w_{2} \\
     w_{3} & w_{4}
  \end{bmatrix}
  `}</Latex
    >. A filter is sometimes called a <Highlight>feature detector</Highlight>.
    That name implies that a specific filter learns to detect a particular
    feature. For example a filter might be able to detect horizontal edges. Once
    the filter learned how to detect those edges, we should be able to detect
    edges independent of the location on the image. Or taken this a step
    further: when we use the same weights to detect an object in an image, it
    should not matter if the object is in the upper left corner or in the middle
    of the image. This is called <Highlight>translation-invariance</Highlight>.
  </p>
  <p>
    The image that is produced by a filter is called a <Highlight
      >feature map</Highlight
    >. Essentially a convolutional operation uses a filter to map an input image
    to an output image that highlights the features that are encoded in the
    filter. In the example below the input image and the kernel (yellow values)
    have pixel values of -1, 0 or 1. The convolutional operation produces
    positive values when a sufficient amount of either positive or negative
    numbers overlap. In our case the filter and the image only sufficiently
    overlap on the right edge. Remember that we are most likely going to apply a
    ReLU non-linearity, which means that most of those numbers are going to be
    set to 0. Different filters would generate different types of overlaps and
    thereby focus on different features.
  </p>
  <Convolution
    imageWidth={5}
    imageHeight={5}
    maxWidth={500}
    kernel={3}
    imageNumbers={[
      [0, 1, 0, 0, 1],
      [1, 1, 0, 0, 1],
      [0, -1, 0, 0, 1],
      [0, 0, -1, -1, 1],
      [0, 1, 0, -1, -1],
    ]}
    kernelNumbers={[
      [-1, 0, 1],
      [-1, 1, 1],
      [0, 0, 1],
    ]}
    showOutput={true}
    showNumbers={true}
  />
  <p>
    Using the same image, but a different filter produces a feature map, that is
    hightlighted in the upper edge.
  </p>
  <Convolution
    imageWidth={5}
    imageHeight={5}
    maxWidth={500}
    kernel={3}
    imageNumbers={[
      [0, 1, 0, 0, 1],
      [1, 1, 0, 0, 1],
      [0, -1, 0, 0, 1],
      [0, 0, -1, -1, 1],
      [0, 1, 0, -1, -1],
    ]}
    kernelNumbers={[
      [1, 0, -1],
      [1, 1, -1],
      [-1, 0, 1],
    ]}
    showOutput={true}
    showNumbers={true}
  />
  <p>
    To allow the convolution neural network to learn several features
    simultaneously, a convolution layer learns several filters with different
    weights and thereby produces several filter maps. The result of a
    convolutional layer is therefore not a single 2d image, but a 3d cube.
  </p>
  <Convolution
    imageWidth={5}
    imageHeight={5}
    maxWidth={500}
    kernel={3}
    numFilters={5}
    showOutput={true}
  />
  <p>
    So far we have dealt with greyscale images in convolution operations, but
    more interesting problems will have 3 channels (red, green, blue). When we
    are dealing with several channels as inputs, our filters gain a channel
    dimension as well. That means that each neuron attends to a 3 dimensional
    receptive field. Below for example the receptive field is 3x3x3 which in
    turn requires a filter with 27 weights.
  </p>
  <Convolution
    imageWidth={5}
    imageHeight={5}
    maxWidth={500}
    kernel={3}
    numChannels={3}
    showOutput={true}
  />
  <p>
    The dimensionality of filters or better said the dimensionality of the
    weight matrix in a convolution layer is (N, C, W, H).
  </p>
  <ul>
    <li>N is the number of feature maps we want to produce</li>
    <li>C is the number of channels of the input image</li>
    <li>W is the width of the image</li>
    <li>H is the height of the image</li>
  </ul>
  <p>
    We can regard the number of produced feature maps as a channel dimension.
    That allows us to stack several convolutional operation.
  </p>
  <div class="separator" />

  <h2>Pooling Layer</h2>
  <p>Pooling - downsampling: lose information but more manageble</p>
  <div class="separator" />

  <h2>The Power Of ConvNets</h2>
  <p>Hierarchy of features</p>
  <p>Why do CNN's work - 1. local features and hierarchical features</p>
  <p>
    We extact features, attention field grows, local features to global features
  </p>
  <p>How does the whole process look like</p>
  <div class="separator" />

  <h2>Backpropagation</h2>
  <p>How does backprop work?</p>
  <div class="separator" />
</Container>
