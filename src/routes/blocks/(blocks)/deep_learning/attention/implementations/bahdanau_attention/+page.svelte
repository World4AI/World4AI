<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/attention/bahdanau.ipynb";

  const url = notebookUrl;
  const fileName = "attention\\bahdanau.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Bahdanau Attention</title>
  <meta
    name="description"
    content="In this section we implement Bahdanau attention in PyTorch."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
