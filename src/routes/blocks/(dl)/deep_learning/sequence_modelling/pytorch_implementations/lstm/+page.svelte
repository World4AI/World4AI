<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/sequence_modelling/lstm.ipynb";

  const url = notebookUrl;
  const fileName = "sequence_modelling\\lstm.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch LSTM</title>
  <meta
    name="description"
    content="An LSTM module in PyTorch does not differ in its interface from the vanilla rnn for the most part, but the layer returns three values instead of just 2: outputs, hidden states and cell states"
  />
</svelte:head>

<JupyterNB {url} {fileName} />
