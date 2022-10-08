<script>
  import Container from "$lib/Container.svelte";
</script>
<h1>VQ-VAE</h1>
<div class="separator"></div>

<Container>

  <p>Encoder network outputs discrete codes</p>
  <p>define latent embedding space e</p>
  <p>Discrete latent variables z are calculated by nearest neighbour lookup to e</p>
  <p>the corresponding e is the input to decoder</p>
  <p>Image</p>
  <p>No real gradient, so copy gradients from decoder inputs to decoder outputs (straight through gradients)</p>
  <p>Loss: reconstruction loss + vq_loss + beta * commitment loss </p>
  <p>vq_loss: move embeddings to encoder outptus</p>
  <p>commitment loss: make sure encoder commits to an embedding and its output does not grow </p>
  <p>The prior is learned, rather than static, autoregressive (PixelCNN)</p>
  <p>vq-vae2</p>
</Container>
