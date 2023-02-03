<script>
  import { getContext } from "svelte";

  export let data = [];
  export let strokeDashArray = "none";
  export let color = "black";
  export let stroke = "1";
  export let isClosed = false;

  const xScale = getContext("xScale");
  const yScale = getContext("yScale");

  let path;
  $: {
    path = `M${data.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join("L")}`;
    if (isClosed){
      path += ' Z';
    }
  }
</script>

{#if data.length > 0}
  <path
    d={path}
    stroke={color}
    stroke-width={stroke}
    stroke-dasharray={strokeDashArray}
  />
{/if}

<style>
  path {
    fill: none;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
</style>
