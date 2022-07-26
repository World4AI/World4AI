<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/pytorch_basics/modules_optimizers_losses.ipynb";

  const url = notebookUrl;
  const fileName = "pytorch_basics\\modueles_optimizers_losses.ipynb";
</script>

<svelte:head>
  <title
    >World4AI | Deep Learning | PyTorch Modules, Optimizers and Losses</title
  >
  <meta
    name="description"
    content="PyTorch provides classes like Modules, Optimizers and Losses to facilitate the training and evaluation process of deep learning."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
