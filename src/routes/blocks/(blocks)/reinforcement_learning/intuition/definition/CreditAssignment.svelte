<script>
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

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
</script>

<svg viewBox="0 0 {width} {height}">
  {#each sequence as { }, i}
    {#each sequence as { }, k}
      {#if k <= i}
        <Arrow
          strokeWidth={0.7}
          dashed={true}
          strokeDashArray="6 6"
          showMarker={false}
          moving={true}
          data={[
            {
              x: margin + i * (boxSize + gap) + boxSize / 2,
              y: margin + boxSize,
            },
            {
              x: margin + k * (boxSize + gap) + boxSize / 2,
              y: height - boxSize - boxSize / 2,
            },
          ]}
        />
      {/if}
    {/each}
  {/each}
  {#each sequence as { d, _ }, i}
    <Block
      x={margin + i * (boxSize + gap) + boxSize / 2}
      y={margin + boxSize / 2}
      width={boxSize}
      height={boxSize}
      text={i + 1}
      fontSize={13}
      class={i < sequence.length - 1 ? "fill-red-400" : "fill-blue-400"}
    />
    <defs>
      <marker
        id="arrowhead"
        markerWidth="10"
        markerHeight="6"
        refX="0"
        refY="3"
        orient="auto"
        class="fill-black"
      >
        <polygon points="0 0, 10 3, 0 6" />
      </marker>
    </defs>
    <!-- Circles containing the actions -->
    <circle
      class="fill-slate-200 stroke-black"
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
      stroke-width="0.5"
      marker-end="url(#arrowhead)"
      class="stroke-black"
    />
  {/each}
</svg>
