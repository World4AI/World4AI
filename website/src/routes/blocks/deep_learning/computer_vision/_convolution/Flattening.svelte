<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte";
  import { tweened } from "svelte/motion";

  export let image = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
  ];

  let height = 200;
  let width = 1000;
  export let blockSize = 23;
  export let gap = 5;

  const progressX = tweened(0, {
    duration: 400,
  });
  const progressY = tweened(0, {
    duration: 400,
  });

  let flattened = false;
  async function clickHandler() {
    if (!flattened) {
      await progressX.set(1);
      await progressY.set(1);
      flattened = true;
    } else {
      await progressY.set(0);
      await progressX.set(0);
      flattened = false;
    }
  }
</script>

<div class="button-container">
  <StepButton on:click={clickHandler} />
</div>
<SvgContainer maxWidth={"1000px"}>
  <svg viewBox="0 0 {width} {height}">
    {#each image as row, rowIdx}
      <g
        transform="translate({$progressX *
          rowIdx *
          row.length *
          (blockSize + gap)}, {-$progressY * rowIdx * (blockSize + gap)})"
      >
        {#each row as pixel, pixelIdx}
          <rect
            x={pixelIdx * (blockSize + gap)}
            y={rowIdx * (blockSize + gap)}
            width={blockSize}
            height={blockSize}
            fill={pixel === 1 ? "black" : "white"}
            stroke="black"
          />
        {/each}
      </g>
    {/each}
  </svg>
</SvgContainer>

<style>
  .button-container {
    margin-bottom: 20px;
  }
</style>
