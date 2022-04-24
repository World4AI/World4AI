<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import { draw } from "svelte/transition";
  export let inputs = ["X1", "X2", "X3"];
  export let outputs = ["Y1", "Y2", "Y3"];
  export let mapping = [[1], [2], [0]];
  export let inputName = "INPUTS";
  export let outputName = "OUTPUTS";
  let activeInput = 0;

  let xMargin = 1;
  let yMargin = 40;
  let size = 70;
  let gap = 20;

  let width = 300;
  let height = size * inputs.length + gap * (inputs.length - 1) + yMargin * 2;
</script>

<SvgContainer maxWidth={"500px"}>
  <svg viewBox="0 0 {width} {height}">
    <text fill="var(--text-color)" x={0} y={25}>{inputName}</text>
    <text fill="var(--text-color)" x={width - 80} y={25}>{outputName}</text>
    {#each mapping as outputs, i}
      {#if i === activeInput}
        {#each outputs as o}
          <line
            in:draw={{ duration: 400 }}
            x1={xMargin + size / 2}
            y1={yMargin + i * size + i * gap + size / 2}
            x2={width - size - xMargin + size / 2}
            y2={yMargin + o * size + o * gap + size / 2}
            stroke="var(--text-color)"
          />
        {/each}
      {/if}
    {/each}
    {#each inputs as input, i}
      <rect
        class="clickable"
        on:click={() => {
          activeInput = i;
        }}
        stroke="var(--main-color-2)"
        fill="var(--aside-color)"
        x={xMargin}
        y={yMargin + i * size + i * gap}
        width={size}
        height={size}
      />
      <text
        class="events"
        fill="var(--text-color)"
        dominant-baseline="middle"
        text-anchor="middle"
        x={xMargin + size / 2}
        y={yMargin + i * size + i * gap + size / 2}>{input}</text
      >
    {/each}
    {#each outputs as output, o}
      <rect
        stroke="var(--main-color-1)"
        fill="var(--aside-color)"
        x={width - size - xMargin}
        y={yMargin + o * size + o * gap}
        width={size}
        height={size}
      />
      <text
        fill="var(--text-color)"
        dominant-baseline="middle"
        text-anchor="middle"
        x={width - size - xMargin + size / 2}
        y={yMargin + o * size + o * gap + size / 2}>{output}</text
      >
    {/each}
  </svg>
</SvgContainer>

<style>
  .clickable {
    cursor: pointer;
  }
  .events {
    pointer-events: none;
  }
</style>
