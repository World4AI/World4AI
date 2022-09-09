<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/sequence_modelling/rnn.ipynb";

  const url = notebookUrl;
  const fileName = "sequence_modelling\\rnn.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Recurrent Neural Networks</title>
  <meta
    name="description"
    content="A recurrent neural network in PyTorch is an additional module, that can easily be integrated with other with other built-in modules like linear layers."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
