<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/sequence_modelling/embeddings.ipynb";

  const url = notebookUrl;
  const fileName = "sequence_modelling\\embeddings.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Word Embeddings</title>
  <meta
    name="description"
    content="PyTorch implements word embeddings as a lookup table, where the index of a word is treated as the key and embeddings are the values."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
