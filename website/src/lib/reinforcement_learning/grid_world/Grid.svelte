<script>
  import { interpolateHcl } from "d3-interpolate";
  // svg parameters
  export let width = 500;
  export let height = 500;
  export let strokeWidth = 0.1;

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

  //additional input
  export let policy = null;
  export let value = null;
  let actionToDegreeMapping = {
    0: 270,
    1: 0,
    2: 90,
    3: 180,
  };
</script>

<svg
  {width}
  {height}
  version="1.1"
  viewBox="0 0 {width} {height}"
  xmlns="http://www.w3.org/2000/svg"
>
  <g id="grid">
    <!-- Create the cells-->
    {#each cells as cell}
      <!-- cells -->
      <rect
        fill="none"
        stroke-width={strokeWidth}
        stroke="var(--text-color)"
        x={cell.c * colSize}
        y={cell.r * rowSize}
        width={colSize}
        height={rowSize}
      />

      <!-- color coded rewards, depening on how large the reward is -->
      {#if showColoredReward}
        <rect
          fill={cell.reward > 0 ? "var(--main-color-2)" : "var(--main-color-1"}
          x={cell.c * colSize}
          y={cell.r * rowSize}
          width={colSize}
          height={rowSize}
        />
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

      <!-- blocks -->
      {#if cell.type === "block"}
        <rect
          fill="var(--text-color)"
          stroke="black"
          stroke-width="3"
          x={cell.c * colSize + obstaclePadding}
          y={cell.r * rowSize + obstaclePadding}
          width={colSize - obstaclePadding * 2}
          height={rowSize - obstaclePadding * 2}
        />
      {/if}

      <!-- goal -->
      {#if cell.type === "goal"}
        <polygon
          fill="var(--text-color)"
          stroke="black"
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
    {/each}

    <!-- player -->
    {#if player}
      <circle
        cx={player.c * colSize + colSize / 2}
        cy={player.r * rowSize + rowSize / 2}
        r={colSize * 0.25}
        fill="var(--text-color)"
        opacity="0.8"
        stroke="black"
        stroke-width="3"
      />
    {/if}

    {#if policy}
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="0"
          refY="3.5"
          orient="auto"
          fill="var(--main-color-1)"
        >
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
      </defs>
      <g>
        {#each policy as arrow}
          <line
            x1={arrow.c * colSize + colSize / 2}
            y1={arrow.r * rowSize + rowSize / 2}
            x2={arrow.c * colSize + colSize - 38}
            y2={arrow.r * rowSize + rowSize / 2}
            transform="rotate({actionToDegreeMapping[arrow.a]}, {arrow.c *
              colSize +
              colSize / 2}, {arrow.r * rowSize + rowSize / 2})"
            stroke="var(--main-color-1)"
            stroke-width="1"
            marker-end="url(#arrowhead)"
          />
        {/each}
      </g>
    {/if}
  </g>
</svg>
