<script>
  import { getContext } from "svelte";
  export let text = "Title";
  export let fontSize = 20;

  let width = getContext("width");
  export let x = width / 2;
  export let y = fontSize;
</script>

<!-- Add title -->
<text font-size={fontSize} class="title" {x} {y}>{text}</text>

<style>
  text.title {
    text-anchor: middle;
    dominant-baseline: middle;
    font-weight: bold;
    fill: var(--text-color);
  }
</style>
