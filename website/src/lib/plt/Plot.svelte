<script>
  import { scaleLinear } from "d3-scale";
  import { setContext } from "svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";

  export let width = 500;
  export let height = 250;
  export let maxWidth = 1000;
  export let domain = [0, 1];
  export let range = [0, 1];
  export let padding = { top: 40, right: 40, bottom: 40, left: 40 };

  let xScale = scaleLinear()
    .domain([...domain])
    .range([padding.left, width - padding.right]);

  let yScale = scaleLinear()
    .domain([...range])
    .range([height - padding.bottom, padding.top]);

  setContext("width", width);
  setContext("height", height);
  setContext("xScale", xScale);
  setContext("yScale", yScale);
  setContext("domain", domain);
  setContext("range", range);

  //TODO
  function handleClick(e) {
    let rect = e.target.getBoundingClientRect();
    let width = rect.width;
    let height = rect.height;

    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;

    let scaleX = d3
      .scaleLinear()
      .domain([0, width])
      .range([domain[0], domain[1]]);
    let scaleY = d3
      .scaleLinear()
      .domain([0, height])
      .range([range[1], range[0]]);

    x = scaleX(x);
    y = scaleY(y);

    startingX = x;
    startingY = y;
  }
</script>

<SvgContainer maxWidth="{maxWidth}px">
  <svg viewBox="0 0 {width} {height}">
    <slot />
  </svg>
</SvgContainer>

<style>
  svg {
    user-select: none;
  }
</style>
