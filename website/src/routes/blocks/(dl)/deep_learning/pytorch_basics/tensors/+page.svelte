<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/pytorch_basics/tensors.ipynb";

  const url = notebookUrl;
  const fileName = "pytorch_basics\\tensors.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Tensors</title>
  <meta
    name="description"
    content="Deep learning relies on highly efficient linear algebra calculations. In PyTorch the Tensor class is the basic multidimensional matrix data type. "
  />
</svelte:head>

<JupyterNB {url} {fileName} />
