<script>
  export let layers = [4, 7, 5, 1];

  let width = 400;
  let height = 150;
  let padding = 5;
  let radius = 4;

  let neuralNetwork = [];

  layers.forEach((neurons, layer) => {
    neuralNetwork.push([]);
    let cx =
      ((width - radius * 2 - padding * 2) / (layers.length - 1)) * layer +
      radius +
      padding;
    let heightPiece = height / neurons;
    for (let neuron = 0; neuron < neurons; neuron++) {
      let cy = heightPiece * (neuron + 1) - heightPiece / 2;
      let config = { cx, cy };
      neuralNetwork[layer].push(config);
    }
  });
</script>

<svg viewBox="0 0 {width} {height}">
  {#each neuralNetwork as layer, idxLayer}
    {#each layer as node, idxNode}
      <!-- draw connections -->
      {#if idxLayer + 1 < layers.length}
        {#each neuralNetwork[idxLayer + 1] as nextNode}
          <line
            x1={node.cx}
            y1={node.cy}
            x2={nextNode.cx}
            y2={nextNode.cy}
            stroke="var(--text-color)"
            stroke-width="0.1px"
          />
        {/each}
      {/if}

      <!-- draw neurons -->
      <circle
        cx={node.cx}
        cy={node.cy}
        r={radius}
        fill="var(--background-color)"
        stroke="var(--text-color)"
      />
    {/each}
  {/each}
</svg>
