<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/autoencoders/autoencoders.ipynb";

  const url = notebookUrl;
  const fileName = "autoencoders\\autoencoders.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Autoencoders in PyTorch</title>
  <meta
    name="description"
    content="In this notebook we implement the plain vanilla autoencoder using PyTorch and the MNIST dataset."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
