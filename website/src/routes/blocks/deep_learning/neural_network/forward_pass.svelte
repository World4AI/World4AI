<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import ForwardPass from "./_forward/ForwardPass.svelte";
</script>

<h1>Forward Pass</h1>
<div class="separator" />
<Container>
  <p>
    In the forward pass the features in the dataset are processed layer by layer
    until the final (the output) layer, where the outputs from the previous
    layer are used as inputs into the current layer. As the name <Highlight
      >forward pass</Highlight
    > suggests, information flows unidirectionally from the input layer to the output
    layer.
  </p>
  <p>How can we formalize this idea?</p>
</Container>
<ForwardPass />
