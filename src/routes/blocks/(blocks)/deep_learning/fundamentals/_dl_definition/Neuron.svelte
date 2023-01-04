<script>
  import { tweened } from "svelte/motion";
  import { draw } from "svelte/transition";
  import StepButton from "$lib/button/StepButton.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  let showCalculation = false;
  let disabled = false;

  const inputY = tweened(0, {
    duration: 1000,
  });

  const outputY = tweened(0, {
    duration: 1000,
  });

  const opacity = tweened(0, {
    duration: 500,
  });

  function wait(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
  }

  async function simulate() {
    disabled = true;
    await inputY.set(170);
    showCalculation = true;
    await wait(1200);
    await opacity.set(1);
    await outputY.set(160);

    await wait(1000);
    await opacity.set(0, { duration: 0 });
    await outputY.set(0, { duration: 0 });
    await inputY.set(0, { duration: 0 });
    showCalculation = false;
    disabled = false;
  }
</script>

<ButtonContainer>
  <StepButton {disabled} on:click={simulate} />
</ButtonContainer>

<svg version="1.1" viewBox="0 0 400 80" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="DotS" overflow="visible" orient="auto">
      <path
        transform="matrix(.2 0 0 .2 1.48 .2)"
        d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
        fill="context-stroke"
        fill-rule="evenodd"
        stroke="context-stroke"
        stroke-width="1pt"
      />
    </marker>
  </defs>
  <g stroke="#000">
    <g id="neuron" class="fill-w4ai-yellow">
      <rect x="186.67" y="16.667" width="46.667" height="46.667" ry="5.8333" />
      <g id="neuron-lines" stroke-width="1px">
        <path d="m192.5 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m204.17 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m215.83 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m227.5 16.667v-11.667" marker-end="url(#DotS)" />
        <path d="m233.33 22.5h11.667" marker-end="url(#DotS)" />
        <path d="m233.33 34.167h11.667" marker-end="url(#DotS)" />
        <path d="m233.33 45.833h11.667" marker-end="url(#DotS)" />
        <path d="m233.33 57.5h11.667" marker-end="url(#DotS)" />
        <path d="m227.5 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m215.83 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m204.17 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m192.5 63.333v11.667" marker-end="url(#DotS)" />
        <path d="m186.67 57.5h-11.667" marker-end="url(#DotS)" />
        <path d="m186.67 45.833h-11.667" marker-end="url(#DotS)" />
        <path d="m186.67 34.167h-11.667" marker-end="url(#DotS)" />
        <path d="m186.67 22.5h-11.667" marker-end="url(#DotS)" />
      </g>
      <g id="neuron-circles">
        <circle cx="192.5" cy="22.5" r="2.4442" />
        <circle cx="227.51" cy="22.5" r="2.4442" />
        <circle cx="192.5" cy="57.398" r="2.4442" />
        <circle cx="227.51" cy="57.398" r="2.4442" />
      </g>
      {#if showCalculation}
        <g
          id="calculation"
          fill="none"
          stroke-width=".2"
          stroke-dasharray="1"
          stroke-dashoffset="2"
        >
          <path
            in:draw={{ duration: 1500 }}
            d="m187.53 22.453h5.0284v11.844h14.479v-2.9193h23.122v7.5681h3.2477"
          />
          <path
            in:draw={{ duration: 1500 }}
            d="m187.89 34.103v9.2648h14.673v-4.5401h5.6058v3.1798h24.732"
          />
          <path
            in:draw={{ duration: 1500 }}
            d="m187.59 45.949h3.0744v7.1361h16.603v-4.5615h25.747"
          />
          <path
            in:draw={{ duration: 1500 }}
            d="m187.35 57.54h1.1796l1.9881 3.4435h5.2734l2.0004-1.1549h25.294v-6.8485h9.6305"
          />
        </g>
      {/if}
      <g stroke-width=".5">
        <path d="m210 34.167h17.5" />
        <path d="m210 40h17.5" />
        <path d="m210 45.833h17.5" />
        <path d="m192.5 51.667h11.667" />
        <path d="m192.5 48.75h11.667" />
        <path d="m192.5 45.833h11.667" />
      </g>
      <rect x="210" y="51.667" width="11.667" height="5.8333" />
      <g id="large-chip" stroke-width=".5">
        <rect x="207.4" y="19.985" width="13.201" height="5.0296" />
        <g>
          <path d="m209 17.5v2.5" />
          <path d="m211.5 17.5v2.5" />
          <path d="m214 17.5v2.5" />
          <path d="m216.5 17.5v2.5" />
          <path d="m219 17.5v2.5" />
          <path d="m219 25v2.5" />
          <path d="m216.5 25v2.5" />
          <path d="m214 25v2.5" />
          <path d="m211.5 25v2.5" />
          <path d="m209 25v2.5" />
        </g>
      </g>
      <g id="small-chip-left">
        <rect x="190" y="31.667" width="5" height="5" stroke-width=".33333" />
        <g stroke-width=".33333px">
          <path d="m191.67 30v1.6667" />
          <path d="m193.33 30v1.6667" />
          <path d="m188.33 33.333h1.6667" />
          <path d="m188.33 35h1.6667" />
          <path d="m191.67 38.333v-1.6667" />
          <path d="m193.33 38.333v-1.6667" />
          <path d="m196.67 33.333h-1.6667" />
          <path d="m196.67 35h-1.6667" />
        </g>
      </g>
      <g id="small-chip-right">
        <rect x="200" y="36.667" width="5" height="5" stroke-width=".33333" />
        <g stroke-width=".33333px">
          <path d="m201.67 35v1.6667" />
          <path d="m203.33 35v1.6667" />
          <path d="m198.33 38.333h1.6667" />
          <path d="m198.33 40h1.6667" />
          <path d="m201.67 43.333v-1.6667" />
          <path d="m203.33 43.333v-1.6667" />
          <path d="m206.67 38.333h-1.6667" />
          <path d="m206.67 40h-1.6667" />
        </g>
      </g>
    </g>
    <path
      id="output-line"
      d="m241.13 40h158.42"
      fill="none"
      stroke-dasharray="2,4"
      stroke-width=".51116"
    />
    <g id="input-lines" fill="none" stroke-dasharray="2,4" stroke-width=".5">
      <path d="m10 22.5h161.04" />
      <path d="m10 57.5h161.04" />
      <path d="m10 34.167h160" />
      <path d="m10 45.833h160" />
    </g>
    <g id="inputs" fill="var(--main-color-1)" stroke-width=".5">
      <circle cx={5.2 + $inputY} cy="22.5" r="2.25" />
      <circle cx={5.2 + $inputY} cy="34.167" r="2.25" />
      <circle cx={5.2 + $inputY} cy="45.833" r="2.25" />
      <circle cx={5.2 + $inputY} cy="57.5" r="2.25" />
    </g>
    <circle
      id="output"
      cx={237.12 + $outputY}
      cy="40"
      r="2.25"
      fill="var(--main-color-2)"
      stroke-width=".5"
      opacity={$opacity}
    />
  </g>
</svg>
