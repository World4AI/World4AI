<script>
  import Container from "$lib/Container.svelte";
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/saving_loading.ipynb";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\saving_loading.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Saving and Loading in PyTorch</title>
  <meta
    name="description"
    content="Often you want to save your model or some other paramters in order to to restore them at a later point. You could for example continue traininig the model or you could use the pretrained model weights for inference."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
