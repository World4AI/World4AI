<script>
  import { getContext } from "svelte";

  export let text = "";
  export let textColor = "black";

  export let x;
  export let y;
  export let fontSize = 12;

  const xScale = getContext("xScale");
  const yScale = getContext("yScale");

</script>

<text
  font-size={fontSize}
  fill={textColor}
  x={xScale(x)}
  y={yScale(y)}>{text}</text
>

<style>
  text {
    dominant-baseline: middle;
  }
</style>
