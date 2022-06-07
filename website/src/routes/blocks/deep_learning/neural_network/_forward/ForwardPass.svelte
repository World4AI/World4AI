<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";
  import Container from "$lib/Container.svelte";

  export let features = [];
  export let weights = [];
  export let biases = [];
  export let netInputs = [];
  export let activations = [];

  let nodes = [];
  let validNetInputs = netInputs[0];
  let validActivations = activations[0];

  nodes.push(features[0]);
  activations.forEach((a) => {
    nodes.push(a[0]);
  });

  //determines which of the nodes is clicked and thereby active
  $: activeLayerIdx = 1;
  $: activeNodeIdx = 1;

  let width = 500;
  let height = 200;
  let size = 40;

  let xGap = width / weights.length - size / weights.length;
  let yGap = 10;

  //calculate the centers of the node based on layer and node indices
  let nodeCenters = [];
  nodes.forEach((layer, layerIdx) => {
    let layerCenters = [];
    layer.forEach((node, nodeIdx) => {
      let x = 1 + layerIdx * xGap - layerIdx;
      let y =
        height / 2 +
        nodeIdx * size +
        nodeIdx * yGap -
        (layer.length * size + (layer.length - 1) * yGap) / 2;
      layerCenters.push({ x, y });
    });
    nodeCenters.push(layerCenters);
  });

  // calculate angles between the centers
  // this gets important to rotate text for weights
  let angles = [];
  nodeCenters.forEach((layers, layerIdx) => {
    let layerAngles = [];
    if (layerIdx !== 0) {
      let nodeAngles;
      layers.forEach((nodeCenter) => {
        nodeAngles = [];
        nodeCenters[layerIdx - 1].forEach((prevNodeCenter) => {
          let dx = nodeCenter.x - size - prevNodeCenter.x;
          let dy = nodeCenter.y - prevNodeCenter.y;
          let rad = Math.atan2(dy, dx);
          let degrees = (rad * 180) / Math.PI;

          nodeAngles.push(degrees);
        });
        layerAngles.push(nodeAngles);
      });
    }
    angles.push(layerAngles);
  });
</script>

<Container maxWidth="1900px">
  <div class="flex-container">
    <div class="left-container">
      <SvgContainer maxWidth="800px">
        <svg
          version="1.1"
          viewBox="0 0 {width} {height}"
          xmlns="http://www.w3.org/2000/svg"
        >
          <!-- draw connections between nodes -->
          <g id="connections" fill="none" stroke="#000" stroke-dasharray="4,2">
            {#each nodes as layer, layerIdx}
              {#if layerIdx !== 0}
                {#each layer as node, nodeIdx}
                  <g
                    stroke={activeLayerIdx === layerIdx &&
                    activeNodeIdx == nodeIdx
                      ? "var(--main-color-2)"
                      : "black"}
                    stroke-width={activeLayerIdx === layerIdx &&
                    activeNodeIdx == nodeIdx
                      ? 2
                      : 0.2}
                  >
                    {#each nodes[layerIdx - 1] as prevNode, prevNodeIdx}
                      <line
                        x1={nodeCenters[layerIdx - 1][prevNodeIdx].x + size}
                        x2={nodeCenters[layerIdx][nodeIdx].x}
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
          {#each nodes as layer, layerIdx}
            <g>
              {#each layer as node, nodeIdx}
                <rect
                  class={layer.type !== "input" ? "clickable" : ""}
                  on:click={() => {
                    if (layerIdx !== 0) {
                      activeLayerIdx = layerIdx;
                      activeNodeIdx = nodeIdx;
                    }
                  }}
                  fill={layerIdx === 0
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

          {#each nodes as layer, layerIdx}
            {#each layer as node, nodeIdx}
              <text
                dominant-baseline="middle"
                text-anchor="middle"
                x={nodeCenters[layerIdx][nodeIdx].x + size / 2}
                y={nodeCenters[layerIdx][nodeIdx].y + size / 2}
                >{node.toFixed(2)}</text
              >
            {/each}
          {/each}

          <!--Draw weights -->
          {#each weights[activeLayerIdx - 1][activeNodeIdx] as weight, nodeIdx}
            <text
              transform="rotate({angles[activeLayerIdx][activeNodeIdx][
                nodeIdx
              ]}, {nodeCenters[activeLayerIdx - 1][nodeIdx].x +
                size +
                6}, {nodeCenters[activeLayerIdx - 1][nodeIdx].y +
                size / 2 -
                6})"
              class="weight-text"
              x={nodeCenters[activeLayerIdx - 1][nodeIdx].x + size + 6}
              y={nodeCenters[activeLayerIdx - 1][nodeIdx].y + size / 2 - 6}
              >w_{nodeIdx}{": "} {weight.toFixed(2)}</text
            >
          {/each}
          <text x={nodeCenters[activeLayerIdx - 1][0].x + size + 6} y={height}
            >b: {biases[activeLayerIdx - 1][0][activeNodeIdx].toFixed(2)}</text
          >
        </svg>
      </SvgContainer>
    </div>

    <table class="right-container">
      <thead>
        <tr>
          <th>Variable</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {#each weights[activeLayerIdx - 1][activeNodeIdx] as weight, idx}
          <tr>
            <td>Input <Latex>x_{idx}</Latex></td>
            <td>{nodes[activeLayerIdx - 1][idx].toFixed(2)}</td>
          </tr>
          <tr>
            <td>Weight <Latex>w_{idx}</Latex></td>
            <td>{weight.toFixed(2)}</td>
          </tr>
        {/each}
        <tr>
          <td>Bias <Latex>b</Latex></td>
          <td>{biases[activeLayerIdx - 1][0][activeNodeIdx].toFixed(2)}</td>
        </tr>
        <tr>
          <td>Net Input <Latex>z = \sum_m w_j x_j + b</Latex></td>
          <td>{validNetInputs[activeLayerIdx - 1][activeNodeIdx].toFixed(2)}</td
          >
        </tr>
        <tr>
          <td
            >Neuron Output <Latex>{String.raw`a = \dfrac{1}{1 + e^{-z}}`}</Latex
            ></td
          >
          <td
            >{validActivations[activeLayerIdx - 1][activeNodeIdx].toFixed(
              2
            )}</td
          >
        </tr>
      </tbody>
    </table>
  </div>
</Container>

<style>
  .flex-container {
    display: flex;
    flex-direction: row;
  }

  .left-container {
    flex-basis: 1900px;
  }
  .right-container {
    flex-shrink: 1;
    max-width: 800px;
  }

  .weight-text {
    font-size: 14px;
  }

  .clickable {
    cursor: pointer;
  }

  text {
    pointer-events: none;
  }

  table {
    width: 100%;
  }

  th {
    text-transform: uppercase;
  }

  td,
  th {
    border: 1px double var(--text-color);
    padding: 7px;
    text-align: center;
  }
  td {
    font-style: italic;
  }

  @media (max-width: 1000px) {
    .flex-container {
      flex-direction: column;
    }
    .left-container {
      flex-basis: initial;
      margin-top: 30px;
    }
    .right-container {
      margin-top: 30px;
    }
  }
</style>
