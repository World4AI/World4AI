<script>
  import Container from "$lib/Container.svelte";
  import Timeline from "$lib/Timeline.svelte";
  import Highlight from "$lib/Highlight.svelte";

  let timeline = [
    {
      date: "1998",
      title: "LeNet",
      description:
        "The LeNet architecture by Yann LeCun is probably the most fundamental architecture in computer vision. Everything else is just an improvement of this basic design.",
    },
    {
      date: "2012",
      title: "AlexNet",
      description:
        "AlexNet was the first architecture that showed, that deep learning can work well on scale. This is the architecture that startet the current AI spring.",
    },
    {
      date: "2014",
      title: "VGG",
      description:
        "The architecture took a simple architecture similar to AlexNet and LeNet and increased the number of layers to the limit.",
    },
    {
      date: "2014",
      title: "GoogLeNet",
      description: "This architecture introduced the inception block.",
    },
    {
      date: "2015",
      title: "ResNet",
      description: "The architecture introduced skip connections.",
    },
    {
      date: "2017",
      title: "MobileNet",
      description:
        "The architecture decreased the size of the model and made ConvNets more efficient on small devices.",
    },
    {
      date: "2019",
      title: "EfficientNet",
      description:
        "The architecture showed how the resolution, depths and width of a convolutional neural network should be scaled in order to gain an efficient neural network with fewer parameters.",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | CNN Architectures</title>
  <meta
    name="description"
    content="There are many different architectures for convolutional neural networks. Those architectures have improved greatly over time, which resulted in gread success in computer vision."
  />
</svelte:head>

<h1>CNN Architectures</h1>
<div class="separator" />
<Container>
  <p>
    In this section we will continue to focus on image classification, but we
    will introduce more and more advanced architectures which are based on
    convolutional neural networks. Those architectures have improved greatly
    over time and have made computer vision a success story.
  </p>
  <p>
    The approach we are going to take is to cover the papers in order they were
    released.
  </p>
  <Timeline {timeline} />
  <p>
    For the most part we have already encountered all buidling blocks that are
    required to implement all the networks from the above list. In essence those
    architectures either attempt to increase the depths of the network in order
    to improve performance or they try to reduce the number of parameters in
    order to speed up training and reduce the number of weghts. Some
    architectures achieved both at the same time.
  </p>

  <p>
    There is obviously a problem that we are going to face, when we try to
    impelement those architectures from scratch.
  </p>
  <p />
  <p class="info">
    We do not have the necessary computational power at our disposal to train
    large models from scratch
  </p>
  <p>
    The amount of data and the number of parameters is just outside of our
    reach. We are therefore going to follow a two-fold strategy. First we will
    implement those architectures from scratch and apply them to a dataset with
    less than 100,000 samples. In the second step we will use transfer learning.
    We have shortly mentioned transfer learning before, but had not the
    opportunity to apply this powerful technique in practice. Transfer learning
    allows you to take already existing pretrained models and to adjust them to
    your needs. The requirements towards computational resources and
    availability of data sinks dramatically once you start to you utilize
    transfer learning.
  </p>
  <p>
    There are generally two ways to utilize transfer learing: <Highlight
      >feature extraction</Highlight
    > and <Highlight>fine-tuning</Highlight>. When we use the pretrained model
    as a feature extractor, we load the model, freeze all weights and replace
    the last couple of layers with the layers that suit our task. As this
    procedure only requires to train a few layers, it tends to be relatively
    fast. When we use fine-tuning, we load the weights, replace the last couple
    of layers, but fune-tune all available weights during the training process.
    There is a potential chance to get better results with fine-tuning, but this
    procedure obviously requires more time.
  </p>
  <p>
    The resoning behind the success of transfer learning is as follows. We have
    mentioned before that the convolutional layers are supposed to learn the
    features of the dataset. It can be argued that if the network has learned to
    recognize edges, colors and higher level features, that those features are
    also useful for other tasks. If the model has learned to classify cats and
    dogs, it should be a relative minor undertaking to adjust the model to
    recognize other animals.
  </p>
  <div class="separator" />
</Container>
