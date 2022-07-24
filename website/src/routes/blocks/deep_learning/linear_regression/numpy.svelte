<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/linear_regression/linear_regression.ipynb";

  const url = notebookUrl;
</script>

<JupyterNB {url} />
