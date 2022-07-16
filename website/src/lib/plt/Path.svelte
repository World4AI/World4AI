<script>
  import { getContext } from "svelte";

  export let data = [];

  const xScale = getContext("xScale");
  const yScale = getContext("yScale");

  let path;
  $: {
    path = `M${data.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join("L")}`;
  }
</script>

<path d={path} />

<style>
  path {
    stroke: var(--text-color);
    fill: none;
    stroke-linejoin: round;
    stroke-linecap: round;
    stroke-width: 1;
  }
</style>
