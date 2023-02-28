<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
</script>

<svelte:head>
  <title>Transfer Learning - World4AI</title>
  <meta
    name="description"
    content="To train computer vision models on real life tasks requires a lot of computational power and data, whch might not be available. Transfer learning allows us to take a pretrained model and to tune it for our purposes. Transfer learning often workes even if we have a lower end computer and a few samles available."
  />
</svelte:head>

<h1>Transfer Learning</h1>
<div class="separator" />
<Container>
  <p>
    So far we have trained our image classification models from scratch using
    the CIFAR-10 dataset. We did that to introduce different types of models
    that helped to establish deep learning. Often our datasets are even smaller
    and/or we do not have the compute to train a large model from scratch.
  </p>
  <Alert type="info">
    You should utilize transfer learning when you do not have the necessary data
    or computational power at our disposal to train large models from scratch
  </Alert>
  <p>
    <Highlight>Transfer learning</Highlight> allows you to take already existing
    pretrained models and to adjust them to your needs. The requirements towards
    computational resources and availability of data sinks dramatically once you
    start to you utilize transfer learning.
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
