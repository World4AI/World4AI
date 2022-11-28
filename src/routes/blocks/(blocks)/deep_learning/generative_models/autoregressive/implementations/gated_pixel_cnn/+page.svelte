<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/autoregressive_generative/gated_pixel_cnn.ipynb";

  const url = notebookUrl;
  const fileName = "autoregressive_generative\\gated_pixel_cnn.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Gated PixelCNN</title>
  <meta
    name="description"
    content="In this section we implement the gated PixelCNN architecture using the PyTorch package."
  />
</svelte:head>

<JupyterNB {url} {fileName} />

