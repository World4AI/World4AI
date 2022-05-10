<script>
  import SvgContainer from "$lib/SvgContainer.svelte";

  export let probHead = 0.5;
  export let probTail = 0.5;
</script>

<SvgContainer maxWidth={"300px"}>
  <svg viewBox="0 0 100 50">
    <circle cx="30" cy="15" r="10" />
    <circle cx="70" cy="15" r="10" />
    <text x="30" y="15.5">H</text>
    <text x="70" y="15.5">T</text>
    <rect x="20" y="30" width="20" height="10" />
    <text x="30" y="35.5">{probHead}</text>
    <rect x="60" y="30" width="20" height="10" />
    <text x="70" y="35.5">{probTail}</text>
    <text x="7" y="35.5">Pr:</text>
  </svg>
</SvgContainer>

<style>
  circle {
    stroke: #000;
    fill: var(--main-color-1);
    stroke-width: 0.5px;
  }

  rect {
    fill: var(--main-color-2);
    stroke: #000;
    stroke-width: 0.5px;
  }

  text {
    font-size: 8px;
    dominant-baseline: middle;
    text-anchor: middle;
  }
</style>
