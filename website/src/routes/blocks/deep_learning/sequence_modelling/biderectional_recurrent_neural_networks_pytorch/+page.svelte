<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/sequence_modelling/rnn_biderectional.ipynb";

  const url = notebookUrl;
  const fileName = "sequence_modelling\\rnn_biderectional.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Biderectional Recurrent Neural Networks</title>
  <meta
    name="description"
    content="Turning a recurrent neural network into a biderectional recurrent neural network in PyTorch requires to pass an additional parameter. PyTorch does all the heavy lifting for you."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
