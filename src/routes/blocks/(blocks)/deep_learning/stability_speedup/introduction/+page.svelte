<script>
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>Stability and Speedup - World4AI</title>
  <meta
    name="description"
    content="Over the years several techniques, like new optimizers, skip connections and batch normalization have been developed to combat long or unstable training. This chapter is dedicated to those techniques."
  />
</svelte:head>

<h1>Stability and Speedup</h1>
<div class="separator" />
<Container>
  <p>
    It takes a lot of time and patience to train a neural network. In practice
    you usually want to iterate often to find a good approach to a particular
    problem. But if a single iteration takes hours (or even days) you might not
    be able to solve your task. Especially when you need to train large models
    on a cluster of GPUs, each saved hour of training will additionally
    correspond to monetary savings.
  </p>
  <p>
    In this chapter we will look at different techniques that will either
    improve your training time significantly or allow you to build models that
    are less prone to instability during training.
  </p>
  <p>
    Be aware, that many of the techniques that we will introduce in this chapter
    are not strictly necessary to sovle MNIST, yet they will be essential down
    the line once we start dealing with convolutional neural networks or
    transformer models.
  </p>
  <div class="separator" />
</Container>
