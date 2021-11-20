<script>
  import { draw, fade } from 'svelte/transition';
  export let cx = 0;
  export let cy = 0;
  let r = 15;
  
  export let rewardValue = 1;
  export let degrees = 0;
  export let active = false;

  let markerWidth = 5;
  let markerHeight = 5;
</script>  


<defs>
    <marker id="rewardArrow" markerwidth={markerWidth} markerheight={markerHeight} refX="0" refY={markerHeight/2} orient="auto" fill="var(--text-color)">
        <polygon points="0 0, {markerWidth} {markerHeight/2}, 0 {markerHeight}" />
    </marker>
</defs>
<circle {cx} {cy} {r} stroke="black" fill="var(--text-color)" />
{#if active}
  <circle transition:fade="{{duration:1000}}" {cx} {cy} r={r+5} opacity=0.2 stroke="black" fill="var(--text-color)" />
{/if}
<text dominant-baseline="middle" text-anchor="middle" x={cx} y={cy}>{rewardValue}</text>
<g transform="rotate({degrees} {cx} {cy})">
  <path marker-end=url(#rewardArrow) d="M {cx + r} {cy} h 25" stroke="var(--text-color)" />
</g>
{#if active}
<g transform="rotate({degrees} {cx} {cy})">
  <path in:draw="{{duration: 1000}}" d="M {cx + r} {cy} h 25" stroke="var(--main-color-1)" />
</g>
{/if}
