<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/basics.ipynb";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\basics.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | CNNs in PyTorch</title>
  <meta
    name="description"
    content="To change a fully connected neural network into a convolutional neural network in PyTorch does not require a lot of work. We only need to add a couple of additional modules."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
