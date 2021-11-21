<script>
  import {draw, fade} from 'svelte/transition';
  export let cx;
  export let cy;
  export let fromX;
  export let fromY;
  let r= 15;
  export let actionValue;
  export let active = false;
  export let xText;
  export let yText;
  export let probability;
</script>

<path d="M {fromX} {fromY} L {cx} {cy}" stroke="var(--text-color)" />
{#if active}
  <path in:draw="{{duration: 1000}}" d="M {fromX} {fromY} L {cx} {cy}" stroke-width="1" stroke="var(--main-color-1)" />
  <circle in:fade="{{duration:1000}}" {cx} {cy} r={r+5} stroke='black' opacity=0.2 fill="var(--main-color-2)" />
{/if}
<circle {cx} {cy} {r} stroke='black' fill="var(--main-color-2)" />
<text dominant-baseline="middle" text-anchor="middle" x={cx} y={cy}>{actionValue}</text>
<text stroke="none" 
      fill={active ? "var(--main-color-1)" : "var(--text-color)"} 
      x={xText} 
      y={yText} 
      font-size=10
      dominant-baseline="middle" 
      text-anchor="middle">{probability}</text>
