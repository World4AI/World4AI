<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";

  export let config = [
    {
      classes: "layer-1",
      type: "input",
      nodes: [{ value: 0.5 }, { value: 0.5 }],
    },
    {
      classes: "layer-2",
      type: "hidden",
      nodes: [
        { weights: [1, 1], bias: 0 },
        { weights: [0.5, 0.5], bias: 1 },
        { weights: [0.5, 0.1], bias: 0 },
        { weights: [0.2, 0.3], bias: 1 },
      ],
    },
    {
      classes: "layer-3",
      type: "hidden",
      nodes: [
        { weights: [1, 0.2, 0.1, 1], bias: 1 },
        { weights: [0.2, 0.1, 0.3, 0.2], bias: 1 },
      ],
    },
    {
      classes: "layer-4",
      type: "output",
      nodes: [{ weights: [1, 1], bias: 1 }],
    },
  ];

  //determines which of the nodes is clicked and thereby active
  let activeLayerIdx = 2;
  let activeNodeIdx = 0;

  let width = 500;
  let height = 200;
  let size = 35;

  let xGap = width / (config.length - 1) - size / (config.length - 1);
  let yGap = 15;

  //calculate the centers of the node based on layer and node indeces
  let nodeCenters = [];
  config.forEach((layer, layerIdx) => {
    let layerCenters = [];
    layer.nodes.forEach((node, nodeIdx) => {
      let x = 1 + layerIdx * xGap - layerIdx;
      let y =
        height / 2 +
        nodeIdx * size +
        nodeIdx * yGap -
        (layer.nodes.length * size + (layer.nodes.length - 1) * yGap) / 2;

      layerCenters.push({ x, y });
    });
    nodeCenters.push(layerCenters);
  });

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  //feedforward calculations
  const values = [];
  function calculateOutputs() {
    config.forEach((layer, layerIdx) => {
      let layerValues = [];
      if (layer.type === "input") {
        layer.nodes.forEach((node) => {
          layerValues.push(node.value);
        });
      } else {
        layer.nodes.forEach((node) => {
          let z = node.bias;
          for (let i = 0; i < node.weights.length; i++) {
            z += node.weights[i] * values[layerIdx - 1][i];
          }
          let a = sigmoid(z);
          layerValues.push(a);
        });
      }
      values.push(layerValues);
    });
  }

  calculateOutputs();
</script>

<SvgContainer maxWidth="700px">
  <svg
    version="1.1"
    viewBox="0 0 {width} {height}"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- draw connections between nodes -->
    <g id="connections" fill="none" stroke="#000" stroke-dasharray="4,2">
      {#each config as layer, layerIdx}
        {#if layer.type !== "input"}
          {#each layer.nodes as node, nodeIdx}
            <g
              stroke={activeLayerIdx === layerIdx && activeNodeIdx == nodeIdx
                ? "var(--main-color-2)"
                : "black"}
              stroke-width={activeLayerIdx === layerIdx &&
              activeNodeIdx == nodeIdx
                ? 4
                : 0.2}
            >
              {#each config[layerIdx - 1].nodes as prevNode, prevNodeIdx}
                <line
                  x1={nodeCenters[layerIdx - 1][prevNodeIdx].x + size / 2}
                  x2={nodeCenters[layerIdx][nodeIdx].x + size / 2}
                  y1={nodeCenters[layerIdx - 1][prevNodeIdx].y + size / 2}
                  y2={nodeCenters[layerIdx][nodeIdx].y + size / 2}
                />
              {/each}
            </g>
          {/each}
        {/if}
      {/each}
    </g>
    <!-- Draw Nodes -->
    {#each config as layer, layerIdx}
      <g class={layer.classes}>
        {#each layer.nodes as node, nodeIdx}
          <rect
            fill={layer.type === "input"
              ? "var(--main-color-1)"
              : layerIdx === activeLayerIdx && nodeIdx === activeNodeIdx
              ? "var(--main-color-3)"
              : "var(--main-color-4)"}
            stroke="var(--text-color)"
            x={nodeCenters[layerIdx][nodeIdx].x}
            y={nodeCenters[layerIdx][nodeIdx].y}
            width={size}
            height={size}
          />
        {/each}
      </g>
    {/each}
  </svg>
  <!-- Draw Additional Information for the Active Node -->
  <div class="parameters yellow">
    <div class="flex">
      <div class="left">
        <!-- inputs -->
        {#each values[activeLayerIdx - 1] as input, idx}
          <p><strong>Input</strong> <Latex>x_{idx}</Latex>:</p>
        {/each}
        <!-- weights -->
        {#each config[activeLayerIdx].nodes[activeNodeIdx].weights as weight, idx}
          <p><strong>Weight</strong> <Latex>w_{idx}</Latex>:{weight}</p>
        {/each}
      </div>
      <div class="right">
        <!-- inputs -->
        {#each values[activeLayerIdx - 1] as input, idx}
          <p>{input.toFixed(4)}</p>
        {/each}
      </div>
    </div>
  </div>
</SvgContainer>

<style>
  .center {
    text-align: center;
  }
  .parameters {
    width: 50%;
    padding: 5px 10px;
  }

  div p {
    margin: 0;
    border-bottom: 1px solid black;
  }

  .flex {
    display: flex;
    flex-direction: row;
  }

  .left {
    flex-grow: 1;
    margin-right: 20px;
  }

  .right {
    flex-basis: 40px;
  }
</style>
