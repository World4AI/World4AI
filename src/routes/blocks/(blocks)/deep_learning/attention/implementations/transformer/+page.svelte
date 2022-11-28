<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/attention/transformer.ipynb";

  const url = notebookUrl;
  const fileName = "attention\\transformer.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Transformer Implementation</title>
  <meta
    name="description"
    content="In this section we implement the original transformer from scratch and train a language model."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
