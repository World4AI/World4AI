<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";

  let width = 250;
  let height = 500;

  let elementHeight = 40;
  let padding = 1;
  let numRows = 7;

  let xAddress = {
    0: padding + elementHeight / 2,
    1: width / 2,
    2: width - (padding + elementHeight / 2),
  };
  let yAddress = {};
  for (let i = 0; i < numRows; i++) {
    yAddress[i] = i * (height / (numRows - 0.45)) + padding + elementHeight / 2;
  }

  const config = [
    {
      type: "input",
      x: padding + elementHeight / 2,
      y: yAddress[0],
      latex: "x_1",
    },
    {
      type: "input",
      x: width - (padding + elementHeight / 2),
      y: yAddress[0],
      latex: "x_2",
    },
    {
      type: "function",
      x: xAddress[0],
      y: yAddress[1],
      latex: "\\ln",
    },
    {
      type: "function",
      x: xAddress[1],
      y: yAddress[1],
      latex: "\\sin",
    },
    {
      type: "function",
      x: xAddress[2],
      y: yAddress[1],
      latex: "\\exp",
    },
    {
      type: "output",
      x: xAddress[0],
      y: yAddress[2],
      latex: "v_1",
    },
    {
      type: "output",
      x: xAddress[1],
      y: yAddress[2],
      latex: "v_2",
    },
    {
      type: "output",
      x: xAddress[2],
      y: yAddress[2],
      latex: "v_3",
    },
    {
      type: "function",
      x: xAddress[1],
      y: yAddress[3],
      latex: "*",
    },
    {
      type: "output",
      x: xAddress[1],
      y: yAddress[4],
      latex: "v_4",
    },
    {
      type: "function",
      x: xAddress[1],
      y: yAddress[5],
      latex: "+",
    },
    {
      type: "output",
      x: xAddress[1],
      y: yAddress[6],
      latex: "v_5",
    },
  ];

  const connections = [
    {
      x1: xAddress[0],
      y1: yAddress[0],
      x2: xAddress[0],
      y2: yAddress[1],
    },
    {
      x1: xAddress[2],
      y1: yAddress[0],
      x2: xAddress[1],
      y2: yAddress[1],
    },
    {
      x1: xAddress[2],
      y1: yAddress[0],
      x2: xAddress[2],
      y2: yAddress[1],
    },
    {
      x1: xAddress[0],
      y1: yAddress[1],
      x2: xAddress[0],
      y2: yAddress[2],
    },
    {
      x1: xAddress[1],
      y1: yAddress[1],
      x2: xAddress[1],
      y2: yAddress[2],
    },
    {
      x1: xAddress[2],
      y1: yAddress[1],
      x2: xAddress[2],
      y2: yAddress[2],
    },
    {
      x1: xAddress[0],
      y1: yAddress[2],
      x2: xAddress[1],
      y2: yAddress[3],
    },
    {
      x1: xAddress[1],
      y1: yAddress[2],
      x2: xAddress[1],
      y2: yAddress[3],
    },
    {
      x1: xAddress[1],
      y1: yAddress[3],
      x2: xAddress[1],
      y2: yAddress[4],
    },
    {
      x1: xAddress[1],
      y1: yAddress[4],
      x2: xAddress[1],
      y2: yAddress[5],
    },
    {
      x1: xAddress[2],
      y1: yAddress[2],
      x2: xAddress[1],
      y2: yAddress[5],
    },
    {
      x1: xAddress[1],
      y1: yAddress[5],
      x2: xAddress[1],
      y2: yAddress[6],
    },
  ];
</script>

<SvgContainer maxWidth="300px">
  <svg viewBox="0 0 {width} {height}">
    {#each connections as connection}
      <line
        x1={connection.x1}
        y1={connection.y1}
        x2={connection.x2}
        y2={connection.y2}
        stroke="black"
      />
    {/each}

    {#each config as element}
      {#if element.type === "input"}
        <circle
          cx={element.x}
          cy={element.y}
          r={elementHeight / 2}
          fill="var(--main-color-3)"
          stroke="var(--text-color)"
        />
      {:else if element.type === "function" || element.type === "output"}
        <rect
          x={element.x - elementHeight / 2}
          y={element.y - elementHeight / 2}
          width={elementHeight}
          height={elementHeight}
          fill={element.type === "function"
            ? "var(--main-color-4)"
            : "var(--main-color-1)"}
          stroke="var(--text-color)"
        />
      {/if}
      <foreignObject
        x={element.x - elementHeight / 2}
        y={element.y - elementHeight / 2}
        width={elementHeight}
        height={elementHeight}
      >
        <div>
          <Latex>{element.latex}</Latex>
        </div>
      </foreignObject>
    {/each}
  </svg>
</SvgContainer>

<style>
  div {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 40px;
  }
</style>
