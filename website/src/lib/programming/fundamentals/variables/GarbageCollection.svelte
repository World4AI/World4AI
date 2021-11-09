<script>
    import {onMount} from 'svelte';
    import { draw } from 'svelte/transition';
    let frame = 0;    
    onMount(() => {
        let interval = setInterval(() => {
            frame+=1;
            if (frame >= 8) {
                frame = 0;
            }
        }, 1500);

        return () => clearInterval(interval);
    })
</script>

<svg width="500" height="500" version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <marker id="marker4005" overflow="visible" orient="auto">
   <path transform="scale(.8)" d="m5.77 0-8.65 5v-10z" fill-rule="evenodd" stroke="var(--main-color)" stroke-width="1pt"/>
  </marker>
 </defs>
 <g fill="var(--text-color)" stroke="var(--text-color)">
  {#if frame > 0 && frame < 7}
  <g transform="translate(-142.03 -75)">
   <text id="environment-text" x="303.05814" y="135.93553" font-family="sans-serif" font-size="29.333px" style="line-height:1.25" xml:space="preserve"><tspan x="303.05814" y="135.93553" font-family="sans-serif" font-size="29.333px">Hello World!</tspan></text>
   <rect x="284.06" y="100" width="215.94" height="50" fill="none" stroke-linejoin="round"/>
  </g>
  {/if}
  <g transform="translate(23.286 170)">
   <rect x="2.8493" y="56.028" width="154.3" height="47.944" fill="none" stroke-linejoin="round"/>
   <text x="12.868568" y="90.090477" font-family="sans-serif" font-size="29.333px" style="line-height:1.25" xml:space="preserve"><tspan x="12.868568" y="90.090477" font-family="sans-serif" font-size="29.333px">sentence</tspan></text>
  </g>
  <g transform="translate(-142.03 165)">
   <g>
    {#if frame > 3}
    <g transform="translate(-30 160)">
     <text x="305.21442" y="135.93553" font-family="sans-serif" font-size="29.333px" style="line-height:1.25" xml:space="preserve"><tspan x="305.21442" y="135.93553" font-family="sans-serif" font-size="29.333px">Goodbye World!</tspan></text>
     <rect x="284.06" y="100" width="275.94" height="50" fill="none" stroke-linejoin="round"/>
    </g>
    {/if}
   </g>
  </g>
  {#if frame > 3}
    <path in:draw="{{duration: 1000}}" d="m100 280 150 130" fill="none" marker-end="url(#marker4005)" stroke-dasharray="8, 2, 1, 2"/>
  {:else if frame > 0 && frame < 4}
    <path in:draw="{{duration: 1000}}" d="m100.02 219.37 139.98-129.25" fill="none" marker-end="url(#marker4005)" stroke-dasharray="8, 2, 1, 2"/>
  {/if}
 </g>
</svg>
