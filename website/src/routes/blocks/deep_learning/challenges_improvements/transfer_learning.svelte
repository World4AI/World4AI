<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Transfer Learning</title>
  <meta
    name="description"
    content="Transfer learning allows us to take a pretrained model and to tune the model to our needs."
  />
</svelte:head>

<h1>Transfer Learning</h1>
<div class="separator" />
<Container>
  <p>
    Chances are, you will not have access to an expensive GPU cluster and you
    will have only a limited amount of data, but you need both in order to train
    modern deep neural networks from scratch. It is very unlikey though, that
    you truly need to train a model from scratch. Instead you should utilize a
    technique called <Highlight>transfer learning</Highlight> whenever possible.
    Transfer learning allows you to take already existing pretrained models and to
    adjust them to your needs. The requirements towards computational resources and
    availability of data sinks dramatically once you start to you utilize transfer
    learning.
  </p>
  <p>
    Models are trained for specific tasks. If you pick the model that was
    trained to classify cats vs dogs, you might achieve relatively good results
    classifying different types of cars. Yet when you pick a model that has seen
    cars in the training process, you will most likely get even better results.
    You should pick the model that is closest to your task at hand.
  </p>
  <p>
    For the time being we will not discuss what models are out there in the
    wild. Once we start covering computer vision and natural language processing
    we will go into more depth. For now let us shortly mention how the process
    of transfer learning actually works.
  </p>
  <p>
    There are generally two ways to utilize transfer learing: <Highlight
      >feature extraction</Highlight
    > and <Highlight>fine-tuning</Highlight>. When we use the pretrained model
    as a feature extractor, we load the model, freeze all weights and replace
    the last couple of layers with the layers that suit our task. As this
    procedure only requires us to train a few layers, it tends to be relatively
    fast. When we use fine-tuning, we load the weights, replace the last couple
    of layers, but fune-tune all available weights during the training process.
    There is a potential chance to get better results with fine-tuning, but this
    procedure obviously requires more time. In practice you will most likely try
    out different approaches and observe what works best.
  </p>
  <div class="separator" />
</Container>
