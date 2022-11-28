<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/autoregressive_generative/pixel_rnn.ipynb";

  const url = notebookUrl;
  const fileName = "autoregressive_generative\\pixel_rnn.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | RowLSTM</title>
  <meta
    name="description"
    content="In this section we implement the RowLSTM architecture from the PixelRNN paper using the PyTorch package."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
