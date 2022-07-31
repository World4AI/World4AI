<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/overfitting/overfitting.ipynb";

  const url = notebookUrl;
  const fileName = "overfitting\\overfitting.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Overfitting</title>
  <meta
    name="description"
    content="PyTorch provides a lot of tools out of the box to deal with overfitting."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
