<script>
  import SvgContainer from "$lib/SvgContainer.svelte";

  export let probHead = 0.5;
  export let probTail = 0.5;
  export let showBits = false;

  let height;
  if (showBits === true) {
    height = 70;
  } else {
    height = 45;
  }
  export let bitsHeads = "1 0";
  export let bitsTails = "1 0 1";
</script>

<SvgContainer maxWidth={"300px"}>
  <svg viewBox="0 0 100 {height}">
    <!-- Heads/Tails -->
    <circle cx="30" cy="15" r="10" />
    <circle cx="70" cy="15" r="10" />
    <text x="30" y="15.5">H</text>
    <text x="70" y="15.5">T</text>
    <!-- Probability -->
    <rect class="probability" x="20" y="30" width="20" height="10" />
    <rect class="probability" x="60" y="30" width="20" height="10" />
    <text x="6" y="35.5">Pr:</text>
    <text x="30" y="35.5">{probHead}</text>
    <text x="70" y="35.5">{probTail}</text>
    <!-- Encoding of information -->
    {#if showBits}
      <rect class="encoding" x="20" y="50" width="20" height="10" />
      <rect class="encoding" x="60" y="50" width="20" height="10" />
      <text x="9" y="55.5">Bits:</text>
      <text x="30" y="55.5">{bitsHeads}</text>
      <text x="70" y="55.5">{bitsTails}</text>
    {/if}
  </svg>
</SvgContainer>

<style>
  circle {
    stroke: #000;
    fill: var(--main-color-1);
    stroke-width: 0.5px;
  }

  rect {
    stroke: #000;
    stroke-width: 0.5px;
  }
  .probability {
    fill: var(--main-color-2);
  }

  .encoding {
    fill: var(--main-color-3);
  }

  text {
    font-size: 8px;
    dominant-baseline: middle;
    text-anchor: middle;
  }
</style>
