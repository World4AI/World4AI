<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/attention/bert.ipynb";

  const url = notebookUrl;
  const fileName = "attention\\bert.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | BERT fine-tuning with HuggingFace Transformers</title>
  <meta
    name="description"
    content="In this section we fine-tune BERT with the help of the transformers library from HuggingFace"
  />
</svelte:head>

<JupyterNB {url} {fileName} />
