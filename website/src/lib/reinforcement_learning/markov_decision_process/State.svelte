<script>
 import {fade} from 'svelte/transition';
 export let x;
 export let y;
 export let size;
 export let stateNumber;
 export let active = false; 
 let activeOffset = 5;
</script>

{#if active}
  <rect in:fade="{{duration: 1000}}" stroke="black" fill="var(--main-color-1)" opacity="0.2" x={x-activeOffset} y={y-activeOffset} width={size + activeOffset * 2} height={size + activeOffset * 2} />
{/if}
<rect stroke="black" fill="var(--main-color-1)" {x} {y} width={size} height={size} />
<text stroke="none" fill="var(--background-color)" x={x + size/2} y={y+size/2} dominant-baseline="middle" text-anchor="middle">
 {stateNumber}
</text>
