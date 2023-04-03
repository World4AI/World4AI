<script>
  import { fly } from "svelte/transition";
  import Latex from "$lib/Latex.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import Circle from "$lib/diagram/Circle.svelte";

  export let f;
  export let showReturn = false;

  let step = 0;
  let ret = 0;
  const maxLen = 10;
  let results = [];
  function fillSequence() {
    let result = f();
    if (result.type === "state") {
      step = step + 1;
    }
    if (result.type === "reward") {
      ret += result.value;
    }
    if (results.length >= maxLen) {
      results.shift();
    }
    results = [...results, result];
  }
</script>

<div class="flex justify-center items-center text-lg">
  <PlayButton f={fillSequence} />
  <span class="bg-blue-100 mx-2 px-2"
    >Timestep <Latex>t</Latex>:
    {#key step}
      <span class="inline-block" in:fly={{ y: -20 }}>
        {step}
      </span>
    {/key}
  </span>
  {#if showReturn}
    <span class="bg-green-100 mx-2 px-2"
      >Return <Latex>G_0</Latex>:
      {#key ret}
        <span class="inline-block" in:fly={{ y: -20 }}>
          {ret}
        </span>
      {/key}
    </span>
  {/if}
</div>
<SvgContainer maxWidth={"600px"}>
  <svg viewBox="0 0 500 80">
    {#each results as result, idx}
      <Circle
        x={25 + idx * 50}
        y={40}
        r={20}
        text={result.value}
        fontSize={20}
        class={result.type === "state"
          ? "fill-red-400"
          : result.type === "reward"
          ? "fill-green-100"
          : "fill-purple-100"}
      />
    {/each}
  </svg>
</SvgContainer>
