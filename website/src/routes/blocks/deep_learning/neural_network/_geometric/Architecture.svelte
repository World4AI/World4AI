<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";
  import Container from "$lib/Container.svelte";

  export let sizes = [2, 4, 2, 1];

  let width = 500;
  let height = 200;
  let size = 40;

  let xGap = width / (sizes.length - 1) - size / (sizes.length - 1);
  let yGap = 10;

  //calculate the centers of the node based on layer and node indices
  let nodeCenters = [];
  sizes.forEach((layer, layerIdx) => {
    let layerCenters = [];
    for (let nodeIdx = 0; nodeIdx < layer; nodeIdx++) {
      let x = 1 + layerIdx * xGap - layerIdx;
      let y =
        height / 2 +
        nodeIdx * size +
        nodeIdx * yGap -
        (layer * size + (layer - 1) * yGap) / 2;
      layerCenters.push({ x, y });
    }
    nodeCenters.push(layerCenters);
  });
</script>

<SvgContainer maxWidth="600px">
  <svg
    version="1.1"
    viewBox="0 0 {width} {height}"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- draw connections between nodes -->
    <g id="connections" fill="none" stroke="#000">
      {#each nodeCenters as layer, layerIdx}
        {#if layerIdx !== sizes.length - 1}
          {#each layer as node, nodeIdx}
            <g stroke="black">
              {#each nodeCenters[layerIdx + 1] as nextNode, nextNodeIdx}
                <line
                  class="clickable"
                  stroke-width="1"
                  stroke-dasharray="10 5"
                  stroke="black"
                  x1={nodeCenters[layerIdx][nodeIdx].x + size}
                  x2={nodeCenters[layerIdx + 1][nextNodeIdx].x}
                  y1={nodeCenters[layerIdx][nodeIdx].y + size / 2}
                  y2={nodeCenters[layerIdx + 1][nextNodeIdx].y + size / 2}
                />
              {/each}
            </g>
          {/each}
        {/if}
      {/each}
    </g>

    <!-- Draw Nodes -->
    {#each nodeCenters as layer, layerIdx}
      <g>
        {#each layer as node, nodeIdx}
          <rect
            fill="var(--main-color-3)"
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
</SvgContainer>

<style>
  .clickable {
    cursor: pointer;
    pointer-events: initial;
  }

  svg,
  svg > * {
    pointer-events: none;
  }
</style>
