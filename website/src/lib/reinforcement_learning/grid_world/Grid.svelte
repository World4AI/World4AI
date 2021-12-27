<script>
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

  export let cells = [];
  export let player;

  // component parameters
  export let showColoredReward = false;
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
  </g>
</svg>
