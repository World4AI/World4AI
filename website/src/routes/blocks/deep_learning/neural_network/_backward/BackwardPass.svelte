<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";
  import Container from "$lib/Container.svelte";

  export let sizes = [2, 4, 2, 1];

  //determines which of the nodes is clicked and thereby active
  $: activeLayerIdx = 0;
  $: activeNodeIdx = 0;
  $: activeNextNodeIdx = 0;

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

<SvgContainer maxWidth="900px">
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
                  stroke-width="2"
                  stroke={(activeLayerIdx === layerIdx &&
                    activeNodeIdx === nodeIdx &&
                    activeNextNodeIdx === nextNodeIdx) ||
                  (activeLayerIdx + 1 === layerIdx &&
                    activeNextNodeIdx === nodeIdx) ||
                  activeLayerIdx + 2 <= layerIdx
                    ? "var(--main-color-1)"
                    : "black"}
                  on:click={() => {
                    activeLayerIdx = layerIdx;
                    activeNodeIdx = nodeIdx;
                    activeNextNodeIdx = nextNodeIdx;
                  }}
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
            fill={layerIdx >= activeLayerIdx + 2 ||
            (layerIdx === activeLayerIdx + 1 &&
              nodeIdx === activeNextNodeIdx) ||
            (layerIdx === activeLayerIdx && nodeIdx === activeNodeIdx)
              ? "var(--main-color-1)"
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

    {#each nodeCenters as layer, layerIdx}
      {#each layer as node, nodeIdx}
        <text
          dominant-baseline="middle"
          text-anchor="middle"
          x={nodeCenters[layerIdx][nodeIdx].x + size / 2}
          y={nodeCenters[layerIdx][nodeIdx].y + size / 2}
          >{layerIdx === 0 ? "x" : "a"}</text
        >
        {#if layerIdx !== 0}
          <text
            font-size="8px"
            dominant-baseline="middle"
            text-anchor="middle"
            x={nodeCenters[layerIdx][nodeIdx].x + 28}
            y={nodeCenters[layerIdx][nodeIdx].y + 10}
          >
            &lt;{layerIdx}&gt;</text
          >
        {/if}
        <text
          font-size="8px"
          dominant-baseline="middle"
          text-anchor="middle"
          x={nodeCenters[layerIdx][nodeIdx].x + 28}
          y={nodeCenters[layerIdx][nodeIdx].y + 30}
        >
          {nodeIdx + 1}</text
        >
      {/each}
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
