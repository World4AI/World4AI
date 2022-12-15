<script>
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";
  import Container from "$lib/Container.svelte";

  let layers = [
    {
      title: "",
      nodes: [
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
      ],
    },
    {
      title: "",
      nodes: [
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
      ],
    },
    {
      title: "",
      nodes: [
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
      ],
    },
    {
      title: "",
      nodes: [
        { value: "", class: "fill-w4ai-red" },
        { value: "", class: "fill-w4ai-red" },
      ],
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Introduction</title>
  <meta
    name="description"
    content="Deep learning is the main contributer of the recent success of machine learning. This block is dedicated to the study of deep learning."
  />
</svelte:head>

<h1>Fundamentals</h1>
<div class="separator" />

<NeuralNetwork {layers} height={220} rectSize={12} verticalGap={10} />
<Container>
  <p>
    This chapter is designed as a gateway into the world of deep learning. We
    will get our feet wet by introducing basic concepts and studying the history
    of deep learning. We will mostly focus on intuition in this chapter and only
    touch mathematics when absolutely necessary.
  </p>
  <div class="separator" />
</Container>
