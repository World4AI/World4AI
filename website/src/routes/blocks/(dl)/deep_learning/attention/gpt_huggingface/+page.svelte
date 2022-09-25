<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/attention/gpt.ipynb";

  const url = notebookUrl;
  const fileName = "attention\\gpt.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch IMDB Sentiment Analysis</title>
  <meta
    name="description"
    content="In this section we use GPT-2 for text generation, based on a starting input sequence. The pipeline function from the HuggingFace transformer library make the inference part exceptionally easy."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
