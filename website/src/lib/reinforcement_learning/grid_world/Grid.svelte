<script>
  // svg parameters
  export let width = 500;
  export let height = 500;
  export let strokeWidth = 1;
  // TODO: Make the calculation dynamic
  let colSize = 100;
  let rowSize = 100;
  //goal
  let goalPadding = 40;
  // obstacles
  let obstaclePadding = 10;

  export let cells = [];
  export let player;
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
      <g
        id="cells"
        fill="none"
        stroke="var(--text-color)"
        stroke-width={strokeWidth}
      >
        <rect
          fill="none"
          x={cell.c * colSize}
          y={cell.r * rowSize}
          width={colSize}
          height={rowSize}
        />
      </g>

      <!-- blocks -->
      <g fill="var(--text-color)" stroke="black" stroke-width="3">
        {#if cell.type === "block"}
          <rect
            x={cell.c * colSize + obstaclePadding}
            y={cell.r * rowSize + obstaclePadding}
            width={colSize - obstaclePadding * 2}
            height={rowSize - obstaclePadding * 2}
          />
        {/if}
      </g>

      <!-- goal -->
      <g fill="var(--text-color)" stroke="black" stroke-width="2">
        {#if cell.type === "goal"}
          <polygon
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
      </g>
    {/each}

    <!-- player -->
    {#if player}
      <g>
        <circle
          cx={player.c * colSize + colSize / 2}
          cy={player.r * rowSize + rowSize / 2}
          r={colSize * 0.25}
          fill="var(--text-color)"
          opacity="0.8"
          stroke="black"
          stroke-width="3"
        />
      </g>
    {/if}
  </g>
</svg>
