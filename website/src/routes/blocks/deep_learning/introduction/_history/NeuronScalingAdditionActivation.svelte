<script>
  import { onMount } from "svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";
  // offset for stroke-dashoffset to emulate movement
  let offset = 0;

  let weight1 = 1;
  let weight2 = 1;
  let input1 = 1;
  let input2 = 1;
  let theta = 0;
  $: scaled1 = weight1 * input1;
  $: scaled2 = weight2 * input2;
  $: sum = scaled1 + scaled2;

  let output;
  $: if (sum <= theta) {
    output = 0;
  } else {
    output = 1;
  }

  onMount(() => {
    let interval = setInterval(() => {
      offset -= 20;
    }, 300);
    return () => {
      clearInterval(interval);
    };
  });
</script>

<svg version="1.1" viewBox="0 0 350 100" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#000">
    <g id="sum-sign">
      <rect
        x="141.37"
        y="43.695"
        width="12.609"
        height="12.609"
        ry="0"
        stroke-linejoin="bevel"
        fill="var(--main-color-4)"
      />
      <path d="m147.67 45v10" stroke-width="1px" />
      <path d="m142.67 50h10" stroke-width="1px" />
    </g>
    <g id="weights" fill="var(--main-color-3)">
      <rect id="weight" x="61" y="5" width="50" height="15" />
      <rect x="61" y="80" width="50" height="15" />
    </g>
    <g id="flow">
      <path
        id="input-top"
        d="m5 12.5h55"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
      />
      <path
        id="input-bot"
        d="m5 87.5h55"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
      />
      <path
        id="summed-output"
        d="m175.64 49.82h59.364"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        stroke={sum >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
        stroke-width={Math.abs(sum)}
      />
      <path
        id="output-top"
        d="m111 10s60-5 60 40"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        stroke={weight1 >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
        stroke-width={Math.abs(weight1)}
      />
      <path
        id="output-bot"
        d="m111 90s60 5 60-40"
        stroke-dasharray="4, 2"
        stroke-dashoffset={offset}
        stroke={weight2 >= 0 ? "var(--main-color-2)" : "var(--main-color-1)"}
        stroke-width={Math.abs(weight2)}
      />
      <path
        id="activated-output"
        d="m255 50h90"
        stroke-dasharray="4,2"
        stroke-dashoffset={offset}
        stroke-width={output}
      />
    </g>
    <rect
      id="theta"
      x="235"
      y="45"
      width="20"
      height="10"
      stroke-linejoin="bevel"
      fill={output === 0 ? "var(--main-color-1)" : "var(--main-color-2)"}
    />
  </g>
</svg>

<div class="separator" />
<Slider min={-5} max={5} step={0.1} bind:value={weight1} />
<Slider min={-5} max={5} step={0.1} bind:value={weight2} />
<Slider min={-5} max={5} step={0.1} bind:value={theta} />
<div class="separator" />

<div class="parameters yellow">
  <div class="flex">
    <div class="left">
      <p><strong>Input</strong> <Latex>x_1</Latex>:</p>
      <p><strong>Weight</strong> <Latex>w_1</Latex>:</p>
      <p><strong>Scaled Value</strong> <Latex>x_1*w_1</Latex>:</p>
      <p><strong>Input</strong> <Latex>x_2</Latex>:</p>
      <p><strong>Weight</strong> <Latex>w_2</Latex>:</p>
      <p><strong>Scaled Value</strong> <Latex>x_2*w_2</Latex>:</p>
      <p>
        <strong>Weighted Sum</strong>
        <Latex>\sum_i x_i w_i</Latex>:
      </p>
      <p><strong>Theta</strong> <Latex>\theta</Latex>:</p>
      <p><strong>Output</strong>:</p>
    </div>
    <div class="right">
      <p><strong>{input1}</strong></p>
      <p><strong>{weight1}</strong></p>
      <p><strong>{scaled1}</strong></p>
      <p><strong>{input2}</strong></p>
      <p><strong>{weight2}</strong></p>
      <p><strong>{scaled2}</strong></p>
      <p><strong>{sum.toFixed(2)}</strong></p>
      <p><strong>{theta.toFixed(2)}</strong></p>
      <p><strong>{output.toFixed(2)}</strong></p>
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
