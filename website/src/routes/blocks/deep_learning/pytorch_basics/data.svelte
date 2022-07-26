<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/pytorch_basics/data.ipynb";

  const url = notebookUrl;
  const fileName = "pytorch_basics\\data.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Dataset and DataLoader</title>
  <meta
    name="description"
    content="Deep learning requires different strategies to work with huge amounts of data. In PyTorch we can utilize the Dataset and DataLoader classes to deal with data."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
