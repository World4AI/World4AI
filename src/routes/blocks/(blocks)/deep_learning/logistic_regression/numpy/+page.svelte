<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/logistic_regression/logistic_regression.ipynb";

  const url = notebookUrl;
  const fileName = "logistic_regression\\logistic_regression.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Linear Regression Numpy</title>
  <meta
    name="description"
    content="Logistic regression can easily and efficiently be implemented in NumPy."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
