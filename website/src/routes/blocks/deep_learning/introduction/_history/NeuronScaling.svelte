<script>
  import { onMount } from "svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";

  // offset for stroke-dashoffset to emulate movement
  let offset = 0;
  // weight that scales the input
  let weight = 1;
  let input = 1;

  onMount(() => {
    let interval = setInterval(() => {
      offset -= 20;
    }, 400);
    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 240 25" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#000">
    <path
      id="input"
      stroke-dashoffset={offset}
      d="m5 12.5h85"
      stroke-dasharray="8, 8"
    />
    <path
      id="output"
      stroke={weight >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
      d="m150 12.5h90"
      stroke-dasharray="8, 8"
      stroke-dashoffset={offset}
      stroke-width={Math.abs(weight)}
    />
    <rect
      id="weight"
      x="95"
      y="5"
      width="50"
      height="15"
      fill="var(--main-color-3)"
    />
  </g>
</svg>
<Slider min={-5} max={5} bind:value={weight} step={0.1} />
<div class="parameters yellow">
  <div class="flex">
    <div class="left">
      <p><strong>Input</strong> <Latex>x</Latex>:</p>
      <p><strong>Weight</strong> <Latex>w</Latex>:</p>
      <p><strong>Scaled Value</strong> <Latex>x*w</Latex>:</p>
    </div>
    <div class="right">
      <p><strong>{input}</strong></p>
      <p><strong>{weight}</strong></p>
      <p><strong>{input * weight}</strong></p>
    </div>
  </div>
</div>

<style>
  .parameters {
    width: 50%;
    padding: 5px 10px;
  }

  div p {
    margin: 0;
    border-bottom: 1px solid black;
  }

  .flex {
    display: flex;
    flex-direction: row;
  }

  .left {
    flex-grow: 1;
    margin-right: 20px;
  }

  .right {
    flex-basis: 40px;
  }
</style>
