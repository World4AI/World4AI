<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/stability_speedup/stability_speedup.ipynb";

  const url = notebookUrl;
  const fileName = "stability_speedup\\stability_speedup.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Stability and Speedup</title>
  <meta
    name="description"
    content="We can implement different optimizers, batch norm and skip connections in a few lines of code in PyTorch."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
