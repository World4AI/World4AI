<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/neural_network/neural_network.ipynb";

  const url = notebookUrl;
  const fileName = "neural_network\\neural_network.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Regression Numpy</title>
  <meta
    name="description"
    content="Backpropagation for relatively easy tasks can be implemented in NumPy."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
