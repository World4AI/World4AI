<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/mixed_precision_training.ipynb";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\mixed_precision_training.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Mixed Precision Training</title>
  <meta
    name="description"
    content="Mixed precision training allows to mix float32 and float16 variables in order to increase training speed and to decrease the vram requirements. The torch.amp module allows us to use automatic mixed precision, which automatically determines which precision for which operation is useful."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
