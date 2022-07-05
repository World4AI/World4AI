<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
  import PlayButton from "$lib/PlayButton.svelte";

  export let width = 500;
  export let height = 250;
  export let numFolds = 10;

  let gap = 2;
  let boxWidth = 100;
  let boxHeight = height / (numFolds + 1) - gap;
  let xOffSet1 = 200;
  let xOffSet2 = 400;

  let activeFold = 0;

  function changeFold() {
    activeFold += 1;
    activeFold %= numFolds;
  }
</script>

<PlayButton on:click={changeFold} />
<br />
<SvgContainer maxWidth="800px">
  <svg viewBox="0 0 {width} {height + gap}">
    <!-- Left -->
    <rect
      x={gap}
      y={gap}
      width={boxWidth}
      height={numFolds * boxHeight + (numFolds - 2) * gap}
      fill="var(--main-color-3)"
    />
    <rect
      x={gap}
      y={gap + numFolds * boxHeight + (numFolds - 2) * gap}
      width={boxWidth}
      height={boxHeight}
      fill="var(--main-color-3)"
    />
    {#each Array(numFolds) as fold, idx}
      <line
        x1={gap + boxWidth}
        x2={xOffSet1}
        y1={idx * 2 + (numFolds * boxHeight + (numFolds - 2) * gap) / 2}
        y2={idx * gap + idx * boxHeight + boxHeight / 2}
        stroke="black"
        stroke-width="0.5px"
        stroke-dasharray="5 5"
      />
    {/each}
    <text x={boxWidth / 2} y={(numFolds * boxHeight + (numFolds - 2) * gap) / 2}
      >Train + Validate</text
    >
    <text
      x={boxWidth / 2}
      y={gap + numFolds * boxHeight + boxHeight / 2 + (numFolds - 2) * gap}
      >Test</text
    >

    <!-- Mid -->
    {#each Array(numFolds) as fold, idx}
      <rect
        x={xOffSet1}
        y={idx * gap + idx * boxHeight}
        width={boxWidth}
        height={boxHeight}
        fill="var(--main-color-4)"
      />
      <text
        x={xOffSet1 + boxWidth / 2}
        y={idx * gap + idx * boxHeight + boxHeight / 2}>Fold Nr. {idx + 1}</text
      >
    {/each}

    <!-- Right -->
    {#each Array(numFolds) as fold, idx}
      <rect
        x={xOffSet2}
        y={idx * gap + idx * boxHeight}
        width={boxWidth}
        height={boxHeight}
        fill={idx === activeFold
          ? "var(--main-color-1)"
          : "var(--main-color-2)"}
      />
      <text
        x={xOffSet2 + boxWidth / 2}
        y={idx * gap + idx * boxHeight + boxHeight / 2}
        >{idx === activeFold ? "Validate" : "Train"}</text
      >
    {/each}
    {#each Array(numFolds) as fold, idx}
      <line
        x1={xOffSet1 + boxWidth}
        x2={xOffSet2}
        y1={idx * gap + idx * boxHeight + boxHeight / 2}
        y2={idx * gap + idx * boxHeight + boxHeight / 2}
        stroke="black"
        stroke-width="0.5px"
        stroke-dasharray="2 2"
      />
    {/each}
  </svg>
</SvgContainer>

<style>
  rect {
    stroke: var(--text-color);
    stroke-width: 0.5px;
  }
  text {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 10px;
    vertical-align: middle;
    display: inline-block;
    font-weight: bold;
  }
</style>
