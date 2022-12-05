<script>
  import { onMount } from "svelte";

  let offset = 0;
  onMount(() => {
    const interval = setInterval(() => {
      offset -= 0.5;
    }, 100);

    return () => {
      clearInterval(interval);
    };
  });

  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";

  export let width = 200;
  export let height = 65;
  export let maxWidth = "500px";
  export let rectSize = 15;
  // size of rect border
  export let border = 1;
  export let verticalGap = 15;
  export let padding = { left: 0, right: 0 };

  export let layers = [];
  /*
    The layers list should contain the following
    {
      title: "Input",
      nodes: [
        { value: "x_1", class: "fill-gray-500" },
        { value: "x_2", class: "fill-gray-500" },
      ],
    },
    {
      title: "Hidden Layer",
      nodes: [
        { value: "a_1", class: "fill-gray-500" },
        { value: "a_2", class: "fill-gray-500" },
      ],
    },
  */

  // calculate the centers of nodes for reusability
  const centers = [];
  let horizontalGap =
    (width - (padding.left + padding.right) - rectSize - border * 2) /
    (layers.length - 1);
  layers.forEach((layer, layerIdx) => {
    let layerCenters = [];
    layer.nodes.forEach((node, nodeIdx) => {
      let x = border + padding.left + layerIdx * horizontalGap;
      let y =
        (height -
          (layer.nodes.length - 1) * verticalGap -
          layer.nodes.length * rectSize) /
          2 +
        nodeIdx * (rectSize + verticalGap);
      layerCenters.push({ x, y });
    });
    centers.push(layerCenters);
  });
</script>

<SvgContainer {maxWidth}>
  <svg viewBox="0 0 {width} {height}">
    {#each layers as layer, layerIdx}
      <!-- Node Title -->
      <text class="title" x={centers[layerIdx][0].x} y="0">{layer.title}</text>
      {#each layer.nodes as node, nodeIdx}
        <!-- Nodes -->
        <rect
          x={centers[layerIdx][nodeIdx].x}
          y={centers[layerIdx][nodeIdx].y}
          width={rectSize}
          height={rectSize}
          stroke-width={border}
          stroke="var(--text-color)"
          class={`${node.class}`}
        />
        <!-- Text Inside Box -->
        <foreignObject
          x={centers[layerIdx][nodeIdx].x}
          y={centers[layerIdx][nodeIdx].y}
          width={rectSize}
          height={rectSize}
        >
          <div><Latex>{node.value}</Latex></div>
        </foreignObject>
      {/each}
    {/each}
    {#each layers as layer, layerIdx}
      {#each layer.nodes as node, nodeIdx}
        {#if layerIdx !== layers.length - 1}
          <!-- Connections -->
          {#each layers[layerIdx + 1].nodes as nextNode, nextNodeIdx}
            <line
              x1={centers[layerIdx][nodeIdx].x + rectSize}
              y1={centers[layerIdx][nodeIdx].y + rectSize / 2}
              x2={centers[layerIdx + 1][nextNodeIdx].x}
              y2={centers[layerIdx + 1][nextNodeIdx].y + rectSize / 2}
              stroke="var(--text-color)"
              stroke-dasharray="2 2"
              stroke-width="0.3"
              stroke-dashoffset={offset}
            />
          {/each}
        {/if}
      {/each}
    {/each}
  </svg>
</SvgContainer>

<style>
  div {
    height: 15px;
    display: flex;
    text-align: center;
    justify-content: center;
    align-items: center;
    font-size: 9px;
  }
  text {
    font-size: 7px;
    dominant-baseline: text-before-edge;
    font-weight: bold;
  }
</style>
