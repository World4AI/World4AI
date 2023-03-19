<script>
  import { interpolateHcl } from "d3-interpolate";
  import SvgContainer from "$lib/SvgContainer.svelte";
  // svg parameters
  export let width = 500;
  export let height = 500;
  export let maxWidth = "500px";
  export let strokeWidth = 0.3;

  // TODO: Make the calculation dynamic
  let colSize = 100;
  let rowSize = 100;
  //goal
  let goalPadding = 40;
  // obstacles
  let obstaclePadding = 10;

  // TODO: Get the colors though javascript
  let interpolate = interpolateHcl("#4EB6D7", "#FF683C");

  export let cells = [];
  export let player;

  // component parameters
  export let showColoredReward = false;
  export let showColoredValues = false;
  export let showOnlyGrid = false;
  export let showReward = false;

  //additional input
  export let policy = null;
  export let valueFunction = null;
  let actionToDegreeMapping = {
    0: 270,
    1: 0,
    2: 90,
    3: 180,
  };

  const round = (n, decimals = 0) =>
    Number(`${Math.round(`${n}e${decimals}`)}e-${decimals}`);
</script>

<SvgContainer {maxWidth}>
  <svg
    version="1.1"
    viewBox="0 0 {width} {height}"
    xmlns="http://www.w3.org/2000/svg"
  >
    <g id="grid">
      <!-- Create the cells-->
      {#each cells as cell}
        <!-- cells -->
        <rect
          stroke-width={strokeWidth}
          x={cell.c * colSize}
          y={cell.r * rowSize}
          width={colSize}
          height={rowSize}
          class="fill-gray-600 stroke-white"
        />

        <!-- color coded rewards, depening on how large the reward is -->
        {#if showColoredReward}
          <rect
            class={cell.reward > 0 ? "fill-blue-500" : "fill-red-500"}
            x={cell.c * colSize}
            y={cell.r * rowSize}
            width={colSize}
            height={rowSize}
          />
        {/if}
        {#if showReward}
          <text
            font-size={25}
            x={cell.c * colSize + colSize / 4}
            y={cell.r * rowSize + colSize / 4}>{cell.reward}</text
          >
        {/if}

        <!-- color coded rewards, depening on how large the reward is -->
        {#if showColoredValues}
          <rect
            fill={interpolate(cell.distance)}
            x={cell.c * colSize}
            y={cell.r * rowSize}
            width={colSize}
            height={rowSize}
          />
        {/if}

        {#if !showOnlyGrid}
          <!-- blocks -->
          {#if cell.type === "block"}
            <rect
              stroke-width="3"
              x={cell.c * colSize + obstaclePadding}
              y={cell.r * rowSize + obstaclePadding}
              width={colSize - obstaclePadding * 2}
              height={rowSize - obstaclePadding * 2}
              class="fill-red-400 stroke-black"
            />
          {/if}

          <!-- goal -->
          {#if cell.type === "goal"}
            <polygon
              stroke="black"
              class="fill-blue-400"
              stroke-width="2"
              points={`${cell.c * colSize + colSize / 2},${
                cell.r * rowSize + goalPadding
              } \
                    ${cell.c * colSize + colSize - goalPadding},${
                cell.r * rowSize + rowSize - goalPadding
              } \
                    ${cell.c * colSize + goalPadding},${
                cell.r * rowSize + rowSize - goalPadding
              }`}
            />
          {/if}

          {#if valueFunction && cell.type != "goal" && cell.type != "block"}
            <text
              fill="black"
              dominant-baseline="middle"
              text-anchor="middle"
              x={cell.c * colSize + colSize / 2}
              y={cell.r * rowSize + rowSize / 2}
              >{round(valueFunction[cell.r][cell.c], 4)}</text
            >
          {/if}
          {#if policy && cell.type != "goal" && cell.type != "block"}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="0"
                refY="3.5"
                orient="auto"
                class="fill-white"
              >
                <polygon points="0 0, 5 3.5, 0 7" />
              </marker>
            </defs>
            <g>
              <line
                x1={(cell.c - 0.1) * colSize + colSize / 2}
                y1={cell.r * rowSize + rowSize / 2}
                x2={cell.c * colSize + colSize - 38}
                y2={cell.r * rowSize + rowSize / 2}
                transform="rotate({actionToDegreeMapping[
                  policy[cell.r][cell.c]
                ]}, {cell.c * colSize + colSize / 2}, {cell.r * rowSize +
                  rowSize / 2})"
                class="stroke-white"
                stroke-width="1.2"
                marker-end="url(#arrowhead)"
              />
            </g>
          {/if}
        {/if}
      {/each}

      <!-- player -->
      {#if player}
        <circle
          cx={player.c * colSize + colSize / 2}
          cy={player.r * rowSize + rowSize / 2}
          r={colSize * 0.25}
          stroke-width="3"
          class="fill-yellow-200 opacity-80 stroke-black"
        />
      {/if}
    </g>
  </svg>
</SvgContainer>

<style>
  text {
    dominant-baseline: middle;
    text-anchor: middle;
  }
</style>
