<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Slider from "$lib/Slider.svelte";

  export let width = 150;
  export let height = 200;
  // max value a gradient can take
  export let maxValue = 3;
  // value or norm to clip to
  export let clipValue = 1;
  // value or norm clipping
  export let type = "value";
  export let boxHeight = 7;

  let weights = [
    { real: 2 },
    { real: 1 },
    { real: 0.5 },
    { real: -2 },
    { real: -3 },
    { real: -1 },
    { real: 3 },
    { real: 2.5 },
    { real: -1.5 },
    { real: -0.5 },
  ];

  // add clipped values
  function recalculateClipped() {
    weights = weights;
    if (type === "value") {
      recalculateClippedValue();
    }
    if (type === "norm") {
      recalculateClippedNorm();
    }
  }

  function recalculateClippedValue() {
    weights.forEach((weight) => {
      if (weight.real > 0) {
        weight.clipped = Math.min(clipValue, weight.real);
      }
      if (weight.real < 0) {
        weight.clipped = Math.max(-clipValue, weight.real);
      }
    });
  }

  function recalculateClippedNorm() {
    let l2 = 0;

    //calc norm
    weights.forEach((weight) => {
      l2 += weight.real ** 2;
    });
    l2 = Math.sqrt(l2);

    weights.forEach((weight) => {
      weight.clipped = (weight.real / l2) * clipValue;
    });
  }

  $: clipValue && recalculateClipped();
</script>

<SvgContainer maxWidth="400px">
  <svg viewBox="0 0 {width} {height}">
    {#each weights as weight, weightIdx}
      <rect
        x={weight.real > 0
          ? width / 2
          : ((Math.abs(weight.real + maxValue) / maxValue) * width) / 2}
        y={(height / weights.length) * weightIdx + 1}
        height={boxHeight}
        width={((width / 2) * Math.abs(weight.real)) / maxValue}
        fill={weight.real > 0 ? "none" : "none"}
        stroke="var(--text-color)"
        stroke-width="0.5"
      />
      <rect
        x={weight.real > 0
          ? width / 2
          : ((Math.abs(weight.clipped + maxValue) / maxValue) * width) / 2}
        y={(height / weights.length) * weightIdx + boxHeight + 2}
        height={boxHeight}
        width={((width / 2) * Math.abs(weight.clipped)) / maxValue}
        fill="var(--main-color-1)"
        stroke="var(--text-color)"
        stroke-width="0.5"
      />
      <line
        x1={width / 2}
        y1="0"
        x2={width / 2}
        y2={height}
        stroke="var(--text-color)"
      />
    {/each}
    <line
      x1={width / 2 - ((width / 2) * clipValue) / maxValue}
      y1="0"
      x2={width / 2 - ((width / 2) * clipValue) / maxValue}
      y2={height}
      stroke="var(--text-color)"
      stroke-width="0.5"
    />
    <line
      x1={width / 2 + ((width / 2) * clipValue) / maxValue}
      y1="0"
      x2={width / 2 + ((width / 2) * clipValue) / maxValue}
      y2={height}
      stroke="var(--text-color)"
      stroke-width="0.5"
    />
  </svg>
</SvgContainer>
<Slider min={0.5} max={2} step={0.1} bind:value={clipValue} />

<style>
</style>
