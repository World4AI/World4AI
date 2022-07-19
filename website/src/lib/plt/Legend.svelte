<script>
  import { getContext } from "svelte";

  export let text = "";
  export let textColor = "black";
  export let legendColor = "black";
  //in the form {x, y}
  export let coordinates = {};
  export let size = 5;

  const xScale = getContext("xScale");
  const yScale = getContext("yScale");

  let boxX = xScale(coordinates.x) - size / 2;
  let boxY = yScale(coordinates.y) - size / 2;
</script>

<rect
  x={boxX}
  y={boxY}
  width={size}
  height={size}
  fill={legendColor}
  stroke="black"
/>
<text
  fill={textColor}
  x={size * 2 + xScale(coordinates.x)}
  y={yScale(coordinates.y)}>{text}</text
>

<style>
  text {
    font-size: 12px;
    dominant-baseline: middle;
  }
</style>
