<script>
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import NeuralNetwork from "$lib/NeuralNetwork.svelte";

  const width = 500;
  const maxWidth = "700px";
  const height = 250;
  const rectSize = 25;

  let layers = [
    {
      title: "",
      nodes: [
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
      ],
    },
    {
      title: "",
      nodes: [
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
      ],
    },
    {
      title: "",
      nodes: [
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
        { value: "", fill: "none" },
      ],
    },
    {
      title: "",
      nodes: [{ value: "", fill: "none" }],
    },
  ];

  let p = 0.2;
  //apply dropout
  function recalculateDropout() {
    layers.forEach((layer, layerIdx) => {
      if (layerIdx !== layers.length - 1) {
        layer.nodes.forEach((node, nodeIdx) => {
          if (Math.random() >= p) {
            layers[layerIdx].nodes[nodeIdx].fill = "none";
          } else {
            layers[layerIdx].nodes[nodeIdx].fill = "var(--main-color-1)";
          }
        });
      }
    });
  }
</script>

<ButtonContainer>
  <PlayButton f={recalculateDropout} delta={800} />
</ButtonContainer>
<NeuralNetwork {width} {maxWidth} {height} {rectSize} {layers} />
