<script>
 import {draw} from 'svelte/transition';
 export let d;
 export let active;
 export let xText;
 export let yText;
 export let probability; 
</script>

<path {d} 
      marker-end=url(#arrowhead)
      stroke="var(--text-color)"
      fill="transparent"
       />
<text stroke="none" 
      fill={active ? "var(--main-color-1)" : "var(--text-color)"} 
      x={xText} 
      y={yText} 
      dominant-baseline="middle" 
      text-anchor="middle">{probability}</text>
<!-- draw active connection -->
{#if active} 
<path in:draw="{{duration: 1000}}" 
      {d} 
      marker-end=url(#arrowhead)
      stroke="var(--main-color-1)"
      fill="transparent"
/>
{/if}
