<script>
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";

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

<h1>Deep Learning</h1>
<div class="separator" />

<NeuralNetwork {layers} height={220} rectSize={12} verticalGap={10} />

<div class="separator" />
<div class="container mx-auto">
  <p>
    Computer vision, natural language processing, generative art, board games
    and many more areas have been taken over by deep learning. Once it became
    clear that we can scale neural networks to improve the performance in a
    particular area, other methods became obsolete. Within a decade from the so
    called ImageNet moment in 2012, deep learning became ubiquitous. Nowadays
    our mobile phones make sure that the pictures you take with your mobile
    phone are as crisp as those taken with a professional camera, recommendation
    systems like those used by Netflix or Spotify do a great job at finding new
    products based on your consumption history and Alexa by Amazon can
    understand your actual speach and trigger actions based on your words. Yet
    all those achievements will pale in comparison to those that will surely
    come in the next decades. Often it feels like we just barely scratched the
    surface of what is actually possible with those systems. This is an exciting
    time to study deep learning and this website will hopefully become a
    stepping stone in your journey towards deep learning mastery.
  </p>
  <p>
    In this block we are going to take a deep dive into deep learning. We will
    study the very first artificial neurons that were developed in the 1950's,
    the progress towards multilayer neural networks with the help of the
    backpropagation algorithm as well as state of the art deep learning.
  </p>
  <p>
    The block is suited for absolute beginners and we do not assume any prior
    knowledge of deep learning, but exposure to linear algebra, probability
    theory and calculus is assumed. We also expect fundamental Python knowledge,
    as theoretical lectures will be accompanied by practical Python books and
    exercises.
  </p>
</div>
<div class="separator" />
