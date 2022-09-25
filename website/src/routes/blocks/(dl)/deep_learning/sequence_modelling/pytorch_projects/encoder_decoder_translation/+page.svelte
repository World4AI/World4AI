<script>
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/sequence_modelling/encoder_decoder.ipynb";

  const url = notebookUrl;
  const fileName = "sequence_modelling\\encoder_decoder.ipynb";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PyTorch Encoder Decoder Translation</title>
  <meta
    name="description"
    content="In this section we implement an encoder-decoder translation model in PyTorch using the english-german text pairs." 
  />
</svelte:head>

<JupyterNB {url} {fileName} />
