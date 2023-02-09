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
  <title>Deep Learning Fundamentals - World4AI</title>
  <meta
    name="description"
    content="This chapter introduces the fundamentals of deep learning. We cover theory of AI in an intuitive fashion and thereby provide a solid foundation required in future chapters."
  />
</svelte:head>

<h1>Fundamentals</h1>
<div class="separator" />

<NeuralNetwork {layers} height={220} rectSize={12} verticalGap={10} />
<Container>
  <p>
    This chapter is designed as a gateway into the world of deep learning. We
    will get our feet wet by introducing basic concepts and studying the history
    of deep learning. Additionally we will talk about development environments,
    deep learning libraries, GPU resources and continuous education in the field
    of deep learning: topics that are fundamental to know if you are an absolute
    beginner.
  </p>
  <div class="separator" />
</Container>
