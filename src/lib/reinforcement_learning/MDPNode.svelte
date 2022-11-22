<script>
  export let x;
  export let row = 0;
  export let node;
  export let distance;
  export let active;

  let stateRadius = 10;
  let actionRadius = 5;
  let horizontalDistance = 60;
  let maxStates = 4;
  let stateSize = 6;
  let y = stateRadius + row * horizontalDistance;
</script>

<g
  style={node.marked ? "outline: thin solid var(--main-color-1)" : ""}
  stroke="var(--text-color)"
  class="impact-group"
>
  <!--connections -->
  {#if node.children.length > 0}
    <line
      x1={x}
      y1={y}
      x2={x - distance}
      y2={stateRadius + (row + 1) * horizontalDistance}
      stroke={node.paths.indexOf(active) != -1 &&
      node.children[0].paths.indexOf(active) != -1
        ? "var(--main-color-1)"
        : "var(--text-color)"}
    />
    <line
      x1={x}
      y1={y}
      x2={x + distance}
      y2={stateRadius + (row + 1) * horizontalDistance}
      stroke={node.paths.indexOf(active) != -1 &&
      node.children[1].paths.indexOf(active) != -1
        ? "var(--main-color-1)"
        : "var(--text-color)"}
    />
  {/if}

  <!--states and actions -->
  <circle
    cx={x}
    cy={y + 1}
    r={node.type === "state" ? stateRadius : actionRadius}
    fill="var(--background-color)"
  />
  <!-- rewards -->
  {#if node.reward}
    <text
      font-size="10"
      fill="var(--text-color)"
      {x}
      y={y + 2}
      dominant-baseline="middle"
      text-anchor="middle">{node.reward}</text
    >
  {/if}
  <!-- draw expectation contribution -->
  {#if node.expectation}
    <!-- rewards -->
    <text
      font-size="9"
      stroke-width="0"
      fill="var(--main-color-2)"
      x={node.type == "action" ? x - 20 : x}
      y={node.type == "action" ? y + actionRadius / 2 : y + stateRadius * 3}
      dominant-baseline="middle"
      text-anchor="middle">{node.expectation.toFixed(2)}</text
    >
  {/if}
  <!-- draw state representation -->
  {#if node.type != "action"}
    <g transform="translate({-(maxStates / 2) * stateSize})">
      {#each Array(maxStates) as _, state}
        <rect
          x={x + state * stateSize}
          y={y + stateRadius + 4}
          width={stateSize}
          height={stateSize}
          fill={state === node.value
            ? "var(--text-color)"
            : "var(--background-color)"}
        />
      {/each}
    </g>
  {/if}
  {#if node.children.length > 0}
    {#each node.children as child, idx}
      <!-- draw probability -->
      <text
        font-size="10"
        stroke={child.type === "action"
          ? "var(--main-color-1)"
          : "var(--text-color)"}
        x={idx === 0 ? x - distance * 1.1 : x + distance * 1.1}
        y={(y + stateRadius + (row + 1) * horizontalDistance) / 2}
        dominant-baseline="middle"
        text-anchor="middle">{child.probability}</text
      >
      <!-- recursive part -->
      <svelte:self
        node={child}
        row={row + 1}
        x={idx === 0 ? x - distance : x + distance}
        distance={distance / 2}
        {active}
      />
    {/each}
  {/if}
</g>
