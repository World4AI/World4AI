<script>
  let height = 100;
  let width = 800;

  let size = 45;
  let distance = 20;
  let leftOffset = 10;
  let y = 20;

  export let trajectory = [
    {
      type: "S",
      subscript: 0,
    },
    {
      type: "A",
      subscript: 0,
    },
    {
      type: "R",
      subscript: 1,
    },
    {
      type: "S",
      subscript: 1,
    },
    {
      type: "A",
      subscript: 1,
    },
    {
      type: "R",
      subscript: 2,
    },
    {
      type: "S",
      subscript: 2,
    },
    {
      type: "A",
      subscript: 2,
    },
    {
      type: "R",
      subscript: 3,
    },
    {
      type: "S",
      subscript: 3,
    },
    {
      type: "...",
      subscript: ".",
    },
    {
      type: "S",
      subscript: "T",
    },
  ];
</script>

<svg viewBox="0 0 {width} {height}">
  {#each trajectory as element, i}
    <rect
      x={leftOffset + i * size + i * distance}
      {y}
      height={size}
      width={size}
      stroke="var(--main-color-1)"
      fill="none"
    />
    <text
      dominant-baseline="middle"
      text-anchor="middle"
      font-size="20px"
      x={leftOffset + i * size + i * distance + size / 2}
      y={y + size / 2}
      stroke="none"
      fill="var(--text-color)"
      >{element.type}
      <tspan
        x={leftOffset + i * size + i * distance + size / 2 + 15}
        y={y + size / 2 + 10}
        font-size="10px">{element.subscript}</tspan
      >
    </text>
  {/each}
</svg>
