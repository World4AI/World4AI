<script>
  import { draw } from "svelte/transition";
  let width = 400;
  let height = 120;
  let sequence = [
    { d: 0, r: "negative" },
    { d: 90, r: "negative" },
    { d: 180, r: "negative" },
    { d: 270, r: "negative" },
    { d: 270, r: "negative" },
    { d: 0, r: "negative" },
    { d: 0, r: "negative" },
    { d: 90, r: "negative" },
    { d: 180, r: "negative" },
    { d: 0, r: "positive" },
  ];
  let margin = 0.5;
  let gap = 15;
  let boxSize = width / sequence.length - gap;

  let activeIdx = 0;
</script>

<svg
  version="1.1"
  viewBox="0 0 {width} {height}"
  xmlns="http://www.w3.org/2000/svg"
>
  {#each sequence as { }, i}
    {#if i === activeIdx}
      {#each sequence as { }, k}
        {#if k <= i}
          <line
            in:draw={{ duration: 400 }}
            x1={margin + i * (boxSize + gap) + boxSize / 2}
            y1={margin + boxSize / 2}
            x2={margin + k * (boxSize + gap) + boxSize / 2}
            y2={height - boxSize}
            stroke="var(--text-color)"
            stroke-width="0.1"
            stroke-dasharray="8,2,1,2"
          />
        {/if}
      {/each}
    {/if}
  {/each}
  {#each sequence as { d, r }, i}
    <rect
      on:click={() => {
        activeIdx = i;
      }}
      class="clickable"
      fill={r === "negative" ? "var(--main-color-1)" : "var(--main-color-2)"}
      stroke="black"
      x={margin + i * (boxSize + gap)}
      y={margin}
      width={boxSize}
      height={boxSize}
    />
    <defs>
      <marker
        id="arrowhead"
        markerWidth="10"
        markerHeight="6"
        refX="0"
        refY="3"
        orient="auto"
        fill="var(--text-color)"
      >
        <polygon points="0 0, 10 3, 0 6" />
      </marker>
    </defs>
    <!-- Circles conaining the action -->
    <circle
      fill="var(--background-color)"
      stroke={i <= activeIdx ? "var(--text-color)" : "black"}
      cx={margin + i * (boxSize + gap) + boxSize / 2}
      cy={height - boxSize}
      r={boxSize / 2}
    />
    <!-- Arrows indicating the actions -->
    <line
      x1={margin + i * (boxSize + gap)}
      y1={height - boxSize}
      x2={margin + i * (boxSize + gap) + boxSize - 5}
      y2={height - boxSize}
      transform="rotate({d}, {margin +
        i * (boxSize + gap) +
        boxSize / 2}, {height - boxSize})"
      stroke="var(--text-color)"
      stroke-width="0.5"
      marker-end="url(#arrowhead)"
    />
  {/each}
</svg>

<style>
  .clickable {
    cursor: pointer;
  }
</style>
