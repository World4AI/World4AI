<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/pytorch_basics/training_loop.ipynb";

  const url = notebookUrl;
  const fileName = "pytorch_basics\\training_loop.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Training Loop</title>
  <meta
    name="description"
    content="The training loop is designed to loop over the data for several epochs and to update the weights and biases."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
