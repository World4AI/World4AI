<script>
  import { onMount } from "svelte";
  /* data should have the format 
  [
    { x: 50, y: 10 },
    { x: 10, y: 50 },
    { x: 20, y: 50 },
  ];
   */
  export let data;
  export let dashed = false;
  export let moving = false;
  export let color = "black";

  let offset = 0;
  onMount(() => {
    if (moving) {
      const interval = setInterval(() => {
        offset -= 0.5;
      }, 200);

      return () => {
        clearInterval(interval);
      };
    }
  });

  let path;
  $: {
    if (data.length > 0) {
      path = `M${data.map((p) => `${p.x},${p.y}`).join("L")}`;
    }
  }
</script>

<marker
  xmlns="http://www.w3.org/2000/svg"
  id="triangle"
  viewBox="0 0 10 10"
  refX="0"
  refY="5"
  markerUnits="strokeWidth"
  markerWidth="4"
  markerHeight="3"
  orient="auto"
  fill={color}
  stroke={color}
>
  <path d="M 0 0 L 10 5 L 0 10 z" />
</marker>

{#if path}
  <path
    d={path}
    stroke={color}
    stroke-dasharray={dashed ? "2 1" : "none"}
    stroke-dashoffset={offset}
    fill="none"
    marker-end="url(#triangle)"
  />
{/if}
