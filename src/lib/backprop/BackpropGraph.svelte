<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Node from "./Node.svelte";
  import { setContext } from "svelte";

  export let graph;
  export let width = 500;
  export let height = 500;

  let levels = {};
  setContext("levels", levels);
</script>

<SvgContainer maxWidth={`${width}px`}>
  <svg viewBox="0 0 {width} {height}">
    <Node root={graph} />
  </svg>
</SvgContainer>
