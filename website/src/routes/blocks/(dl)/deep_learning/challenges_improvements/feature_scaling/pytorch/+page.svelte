<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/feature_scaling/feature_scaling.ipynb";

  const url = notebookUrl;
  const fileName = "feature_scaling\\feature_scaling.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Feature Scaling</title>
  <meta
    name="description"
    content="Feature scaling for images can be implemented in PyTorch using the torchvision package. Implementing feature scaling improves the convergence speed greatly."
  />
</svelte:head>

<JupyterNB {url} {fileName} />
