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
  export let strokeWidth = 1;
  export let strokeDashArray = "2 1";
  export let speed = 200;

  export let showMarker = true;
  export let markerWidth = 4; 
  export let markerHeight = 3;

  let offset = 0;
  onMount(() => {
    if (moving) {
      const interval = setInterval(() => {
        offset -= 0.5;
      }, speed);

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
  markerWidth={markerWidth}
  markerHeight={markerHeight}
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
    stroke-dasharray={dashed ? strokeDashArray : "none"}
    stroke-dashoffset={offset}
    fill="none"
    stroke-width={strokeWidth}
    marker-end={showMarker ? "url(#triangle)" : "none"}
  />
{/if}
