<script>
  import Table from "$lib/Table.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Diagram from "$lib/diagram/Diagram.svelte";
  import Convolution from "../../image_classification/_convolution/Convolution.svelte";

  let header = ["Type", "Input Size", "Output Size"];
  let data = [
    ["Basic Block", "224x224x3", "112x112x64"],
    ["Max Pooling", "112x112x64", "56x56x64"],
    ["Basic Block", "56x56x64", "56x56x64"],
    ["Basic Block", "56x56x64", "56x56x192"],
    ["Max Pooling", "56x56x192", "28x28x192"],
    ["Inception", "28x28x192", "28x28x256"],
    ["Inception", "28x28x256", "28x28x480"],
    ["Max Pooling", "28x28x480", "14x14x480"],
    ["Inception", "14x14x480", "14x14x512"],
    ["Inception", "14x14x512", "14x14x512"],
    ["Inception", "14x14x512", "14x14x512"],
    ["Inception", "14x14x512", "14x14x528"],
    ["Inception", "14x14x528", "14x14x832"],
    ["Max Pooling", "14x14x832", "7x7x832"],
    ["Inception", "7x7x832", "7x7x832"],
    ["Inception", "7x7x832", "7x7x1024"],
    ["Avg. Pooling", "7x7x1024", "1x1x1024"],
    ["Dropout", "-", "-"],
    ["Fully Connected", "1024", "1000"],
    ["Softmax", "1000", "1000"],
  ];

  let references = [
    {
      author:
        "Szegedy, Christian and Wei Liu and Yangqing Jia and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew",
      title: "Going deeper with convolutions",
      journal:
        "2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      year: "2015",
      pages: "1-9",
      volume: "",
      issue: "",
    },
    {
      author: "Lin, M., Chen, Q., & Yan, S.",
      title: "Network in Network",
      journal: "",
      year: "2013",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author:
        "Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, ZB",
      title: "Rethinking the Inception Architecture for Computer Vision",
      journal: "",
      year: "2016",
      pages: "",
      volume: "",
      issue: "",
    },
  ];

  let gap = 1;

  // basic block
  let basicWidth = 90;
  let basicHeight = 100;
  let basicBoxWidth = 80;
  let basicBoxHeight = 20;
  let basicMaxWidth = "200px";

  const basicComponents = [
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight - basicBoxHeight / 2 - gap,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "Conv2d",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicHeight / 2,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "BatchNorm2d",
      color: "var(--main-color-4)",
    },
    {
      type: "block",
      x: basicWidth / 2,
      y: basicBoxHeight / 2 + gap,
      width: basicBoxWidth,
      height: basicBoxHeight,
      text: "ReLU",
      color: "var(--main-color-4)",
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight - basicBoxHeight - gap,
        },
        {
          x: basicWidth / 2,
          y: basicHeight / 2 + basicBoxHeight / 2 + 3,
        },
      ],
    },
    {
      type: "arrow",
      dashed: true,
      moving: true,
      data: [
        {
          x: basicWidth / 2,
          y: basicHeight / 2 - basicBoxHeight / 2,
        },
        {
          x: basicWidth / 2,
          y: basicBoxHeight + 4,
        },
      ],
    },
  ];

  //create diagram for inception block
  let width = 1000;
  let height = 500;
  let boxWidth = 210;
  let boxHeight = 70;
  let maxWidth = "800px";

  const components = [
    // concatenation
    {
      type: "block",
      x: width / 2,
      y: boxHeight / 2 + gap,
      width: boxWidth,
      height: boxHeight,
      color: "none",
      text: "Concatenation",
      fontSize: 25,
    },

    //top layer
    {
      type: "block",
      x: 0 + boxWidth / 2 + gap,
      y: height - boxHeight - height / 2,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-1)",
      text: "1x1 Basic Block",
      fontSize: 25,
    },
    {
      type: "block",
      x: (width / 3) * 1,
      y: height - boxHeight - height / 2,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-2)",
      text: "3x3 Basic Block",
      fontSize: 25,
    },
    {
      type: "block",
      x: (width / 3) * 2,
      y: height - boxHeight - height / 2,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-3)",
      text: "5x5 Basic Block",
      fontSize: 25,
    },
    {
      type: "block",
      x: (width / 3) * 3 - boxWidth / 2 - gap,
      y: height - boxHeight - height / 2,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-4)",
      text: "1x1 Basic Block",
      fontSize: 25,
    },

    // second layer
    {
      type: "block",
      x: (width / 3) * 1,
      y: height - height / 3,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-2)",
      text: "1x1 Basic Block",
      fontSize: 25,
    },
    {
      type: "block",
      x: (width / 3) * 2,
      y: height - height / 3,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-3)",
      text: "1x1 Basic Block",
      fontSize: 25,
    },
    {
      type: "block",
      x: (width / 3) * 3 - boxWidth / 2 - gap,
      y: height - height / 3,
      width: boxWidth,
      height: boxHeight,
      color: "var(--main-color-4)",
      text: "3x3 MaxPool",
      fontSize: 25,
    },

    // input
    {
      type: "block",
      x: width / 2,
      y: height - boxHeight / 2 - gap,
      width: boxWidth,
      height: boxHeight,
      color: "none",
      text: "Input",
      fontSize: 25,
    },

    // arrows bot
    {
      type: "arrow",
      data: [
        {
          x: width / 2,
          y: height - boxHeight,
        },
        {
          x: (width / 3) * 1,
          y: height - height / 3 + boxHeight / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: width / 2,
          y: height - boxHeight,
        },
        {
          x: (width / 3) * 2,
          y: height - height / 3 + boxHeight / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: width / 2 + boxWidth / 2,
          y: height - boxHeight / 2,
        },
        {
          x: (width / 3) * 3 - boxWidth / 2 - gap,
          y: height - boxHeight / 2,
        },
        {
          x: (width / 3) * 3 - boxWidth / 2 - gap,
          y: height - height / 3 + boxHeight / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: width / 2 - boxWidth / 2,
          y: height - boxHeight / 2,
        },
        {
          x: 0 + boxWidth / 2 + gap,
          y: height - boxHeight / 2,
        },
        {
          x: 0 + boxWidth / 2 + gap,
          y: height - boxHeight / 2 - height / 2 + 10,
        },
      ],
    },
    // arrows mid
    {
      type: "arrow",
      data: [
        {
          x: (width / 3) * 1,
          y: height - height / 3 - boxHeight / 2,
        },
        {
          x: (width / 3) * 1,
          y: height - boxHeight / 2 - height / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: (width / 3) * 2,
          y: height - height / 3 - boxHeight / 2,
        },
        {
          x: (width / 3) * 2,
          y: height - boxHeight / 2 - height / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: (width / 3) * 3 - boxWidth / 2 - gap,
          y: height - height / 3 - boxHeight / 2,
        },
        {
          x: (width / 3) * 3 - boxWidth / 2 - gap,
          y: height - boxHeight / 2 - height / 2 + 10,
        },
      ],
    },
    // arrows top
    {
      type: "arrow",
      data: [
        {
          x: 0 + boxWidth / 2 + gap,
          y: height - boxHeight - boxHeight / 2 - height / 2,
        },
        {
          x: 0 + boxWidth / 2 + gap,
          y: boxHeight / 2,
        },
        {
          x: width / 2 - boxWidth / 2 - 10,
          y: boxHeight / 2,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: (width / 3) * 1,
          y: height - boxHeight - boxHeight / 2 - height / 2,
        },
        {
          x: width / 2 - boxWidth / 2,
          y: boxHeight / 2 + boxHeight / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: (width / 3) * 2,
          y: height - boxHeight - boxHeight / 2 - height / 2,
        },
        {
          x: width / 2 + boxWidth / 2,
          y: boxHeight / 2 + boxHeight / 2 + 10,
        },
      ],
    },
    {
      type: "arrow",
      data: [
        {
          x: (width / 3) * 3 - boxWidth / 2 - gap,
          y: height - boxHeight - boxHeight / 2 - height / 2,
        },
        {
          x: (width / 3) * 3 - boxWidth / 2 - gap,
          y: boxHeight / 2,
        },
        {
          x: width / 2 + boxWidth / 2 + 10,
          y: boxHeight / 2,
        },
      ],
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | GoogLeNet</title>
  <meta
    name="description"
    content="The GoogLeNet architecture combines several layers of Inception modules to create a deep convolutional neural network. An inception module simultaneously calculates convolutions with different kernel size using the same input and the results are then concatenated."
  />
</svelte:head>

<h1>GoogLeNet</h1>
<div class="separator" />
<Container>
  <p>
    The name GoogLeNet<InternalLink type={"reference"} id={1} /> was developed, as
    the name suggests, by researchers at Google. But the name is also a reference
    to the original LeNet-5 architecture, a sign of respect for Yann LeCun. GoogLeNet
    achieved a top-5 error rate of 6.67% (VGG achieved 7.32) and won the 2014 ImageNet
    classification challenge.
  </p>

  <p>
    The GoogLeNet network is a specific, 22 layer, realization of the so called
    Inception architecture. This architecture uses an Inception block, a
    multibranch block that applies convolutions of different filter sizes to the
    same input and concatenates the results in the final step. This architecture
    choice removes the need to search for the optimal patch size and allows the
    creation of much deeper neural networks, while being very efficient at the
    same time. In fact the GoogLeNet architecture uses 12x fewer parameters than
    AlexNet.
  </p>

  <p>
    In the very first step we create basic building block that is going to be
    utilized in each convolutional layer. The block constists of a convolutional
    laye with variable filter and feature map size. The convolutiona is followed
    by a batch norm layer and a ReLU activation function. In the original
    implementation batch normalization was not used, instead in order to deal
    with vanishing gradients, the authors implemented several losses along the
    path of the neural network. This approach is very uncommon and we are not
    going to implement these so called auxilary losses. Batch normalization is a
    much simpler approach.
  </p>
  <Diagram
    width={basicWidth}
    height={basicHeight}
    maxWidth={basicMaxWidth}
    components={basicComponents}
  />
  <p>
    The Inception block takes an input from a previous layer and applies
    calculations in 4 different branches, before concatenating the branches in
    the last step.
  </p>
  <Diagram arrowStrokeWidth={3} {width} {height} {maxWidth} {components} />
  <p>
    You will notice that aside from the expected 3x3 convolutions, 5x5
    convolutions and max pooling, there is a 1x1 convolution in each single
    branch. You might suspect that the 1x1 convolution operation produces an
    output, that is equal to the input. If you think that, then your intuition
    is wrong. Remember that the convolution operation is applied to all feature
    maps in the previous layer. While the width and the height after the 1x1
    convolution remain the same, the number of filters can be changed
    arbitrarily. Below for example we take 4 feature maps as input and return
    just one single feature map.
  </p>

  <Convolution
    imageWidth={6}
    imageHeight={6}
    kernel={1}
    showOutput="true"
    numChannels={4}
    numFilters={1}
  />
  <p>
    This operation allows us to reduce the number of feature maps in order to
    save computational power. This is especially relevant for the 3x3 and 5x5
    filters, as those require a lot of weights, when the number of filter grows.
    That means that in the inception block we reduce the number of filters,
    before we apply the 3x3 and 5x5 filters.
  </p>
  <p>
    You should also bear in mind that in each branch the size of the feature
    maps have to match. If they wouldn't, you would not be able to concatenate
    the branches in the last step. The number of channels after the
    concatenation corresponds to the sum of the channels from each branch.
  </p>

  <p>
    The overall GoogLeNet architecture combines many layers of Inception and
    Pooling blocks. You can get the exact parameters either by studying the
    original paper, or by looking at the code below.
  </p>
  <Table {header} {data} />
  <p>
    Aside from the inception blocks, there are more details, that we have not
    observed so far. So far we have used several fully connected layers in the
    classification block. We did that to slowly move from the flattened vector
    to the number of neurons that are used as input into the softmax layer. In
    the GoogLeNet architecture the last pooling layer removes the width and
    length and we use a single fully connected layer, before the sigmoid/softmax
    layer. Such a procedure is quite common nowadays. Fully connected layers
    require many parameters and the approach above avoids unnecessary
    calculations, moving from convolutions to the softmax in a single fully
    connected step. The 1x1 convolutions and the use of an average pooling
    instead of several fully connected layers were popularized by Lin et al. in
    their 2013 paper<InternalLink type={"reference"} id={2} />.
  </p>
  <p>
    Be aware that the architecture we have discussed above is often called
    InceptionV1. Over the years, the paper on BatchNorm was released and applied
    to the Inception architecture and InceptionV2 and InceptionV3<InternalLink
      type={"reference"}
      id={3}
    /> were released. Below we will implement the original Inception (GoogLeNet)
    architecture with batch normalization.
  </p>
  <div class="separator" />
</Container>

<Footer {references} />
