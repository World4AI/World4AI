 <script>
    import { onMount } from 'svelte';

    let offset= 0;
    onMount(() => {
        let interval = setInterval(() => {
            offset+= 20;
        }, 300);

        return () => clearInterval(interval);
    })
</script>

<svg width="500" height="250" version="1.1" viewBox="0 0 500 250" xmlns="http://www.w3.org/2000/svg">
 <defs>
  <marker id="marker4005" overflow="visible" orient="auto">
   <path transform="scale(.8)" d="m5.77 0-8.65 5v-10z" fill-rule="evenodd" stroke="var(--text-color)" stroke-width="1pt"/>
  </marker>
 </defs>
 <g  fill="var(--text-color)" transform="translate(-142.03 -75)">
  <text id="environment-text" x="303.05814" y="135.93553" font-family="sans-serif" font-size="29.333px" style="line-height:1.25" xml:space="preserve"><tspan x="303.05814" y="135.93553" font-family="sans-serif" font-size="29.333px">Hello World!</tspan></text>
  <rect x="284.06" y="100" width="215.94" height="50" fill="none" stroke="var(--text-color)" stroke-linejoin="round"/>
 </g>
 <g transform="translate(23.286 120.56)">
  <rect x="0" y="56.028" width="154.3" height="47.944" fill="none" stroke="var(--text-color)" stroke-linejoin="round"/>
  <text x="12.868568" y="90.090477" font-family="sans-serif" font-size="29.333px" style="line-height:1.25" xml:space="preserve"><tspan x="12.868568" y="90.090477" font-family="sans-serif" font-size="29.333px" fill="var(--text-color)">sentence</tspan></text>
 </g>
 <g transform="translate(28.286 -70.251)">
  <rect x="260" y="246.83" width="182.13" height="47.944" fill="none" stroke="var(--text-color)" stroke-linejoin="round"/>
  <text x="271.40747" y="280.89719" font-family="sans-serif" font-size="29.333px" style="line-height:1.25" xml:space="preserve"><tspan x="271.40747" y="280.89719" font-family="sans-serif" font-size="29.333px" fill="var(--text-color)">sentence_2</tspan></text>
 </g>
 <path d="m100 160 140-80" fill="none" marker-end="url(#marker4005)" stroke="var(--text-color)" stroke-dashoffset={offset} stroke-dasharray="8, 2, 1, 2"/>
 <path d="m390 160-130-80" fill="none" marker-end="url(#marker4005)" stroke="var(--text-color)" stroke-dashoffset={offset} stroke-dasharray="8, 2, 1, 2"/>
</svg>  
