<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/sequence_modelling/sentiment_analysis_imdb.ipynb";

  const url = notebookUrl;
  const fileName = "sequence_modelling\\sentiment_analysis_imdb.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch IMDB Sentiment Analysis</title>
  <meta
    name="description"
    content="In this section we generate a model that can predict the IMDB review sentiment. The IMDB dataset is considered to be the MNIST of sequence modelling. It is a beginner level dataset that contains movie reviews and the corresponding sentiment: positive or negative."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
