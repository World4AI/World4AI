<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/lenet_5.ipynb";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\lenet_5.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | LeNet-5</title>
  <meta
    name="description"
    content="LeNet-5 is one of the oldest and at the same time one of most well known convolutional neural networks architectures."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
