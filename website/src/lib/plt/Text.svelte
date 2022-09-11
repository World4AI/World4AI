<script>
  import { getContext } from "svelte";

  export let text = "";
  export let textColor = "black";

  export let x;
  export let y;
  export let size = 5;

  const xScale = getContext("xScale");
  const yScale = getContext("yScale");

</script>

<text
  fill={textColor}
  x={size * 2 + xScale(x)}
  y={yScale(y)}>{text}</text
>

<style>
  text {
    font-size: 12px;
    dominant-baseline: middle;
  }
</style>
