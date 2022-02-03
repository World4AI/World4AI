<script>
  import MDPNode from "$lib/reinforcement_learning/MDPNode.svelte";

  export let width = 500;
  export let height = 300;
  export let root;
  export let active = null;
</script>

<svg viewBox="0 0 {width} {height}">
  <MDPNode node={root} x={width / 2} distance={width / 4} {active} />
</svg>

<style>
  svg {
    max-width: 1000px;
  }
</style>
