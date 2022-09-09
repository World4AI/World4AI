<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/pytorch_basics/autograd.ipynb";

  const url = notebookUrl;
  const fileName = "pytorch_basics\\autograd.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Autograd</title>
  <meta
    name="description"
    content="Autograd is the automatic differentiation package that is included in PyTorch."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
