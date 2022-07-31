<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/vanishing_exploding_gradients/vanishing_exploding_gradients.ipynb";

  const url = notebookUrl;
  const fileName =
    "vanishing_exploding_gradients\\vanishing_exploding_gradients.ipynb";
</script>

<svelte:head>
  <title
    >World4AI | Deep Learning | PyTorch Vanishing and Exploding Gradients</title
  >
  <meta
    name="description"
    content="PyTorch provides different activation functions, weight initialization schemes and gradient clipping to deal with problematic gradients."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
