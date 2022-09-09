<script>
  import Plot from "$lib/Plot.svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";
  export let p = 0.5;
  let entropy = 0;
  $: if (p === 0 || p === 1) {
    entropy = 0;
  } else {
    entropy = -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  }
  $: pointsData = [{ x: p, y: entropy }];

  let pathsData = [];
  for (let i = 0.001; i < 1; i += 0.001) {
    let x = i;
    let y = -x * Math.log2(x) - (1 - x) * Math.log2(1 - x);
    let data = { x, y };
    pathsData.push(data);
  }
</script>

<Plot
  {pathsData}
  {pointsData}
  config={{
    minX: 0,
    maxX: 1,
    minY: 0,
    maxY: 1.05,
    xLabel: "p(x)",
    yLabel: "Entropy",
    padding: { top: 20, right: 40, bottom: 40, left: 50 },
    xTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
  }}
/>
<div class="parameters yellow">
  <div class="flex">
    <div class="left">
      <p><strong>Probability Heads</strong> <Latex>p(x)</Latex>:</p>
      <p><strong>Probability Tails</strong> <Latex>1-p(x)</Latex>:</p>
      <p><strong>Entropy</strong> <Latex>H(x)</Latex>:</p>
    </div>
    <div class="right">
      <p><strong>{p.toFixed(3)}</strong></p>
      <p><strong>{(1 - p).toFixed(3)}</strong></p>
      <p><strong>{entropy.toFixed(5)}</strong></p>
    </div>
  </div>
</div>
<Slider min={0} max={1} step={0.001} bind:value={p} />

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
