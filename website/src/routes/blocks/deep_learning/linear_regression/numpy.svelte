<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/linear_regression/linear_regression.ipynb";

  const url = notebookUrl;
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Regression Numpy</title>
  <meta
    name="description"
    content="Linear regression can easily and efficiently be implemented in NumPy."
  />
</svelte:head>
<JupyterNB {url} />
