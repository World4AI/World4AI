<script>
  export let width = 1000;
  export let height = 150;
  export let type = 'Integer';
  export let value = 42;
  export let count = 1;
  export let boxHeight = 30;
  export let boxWidth = 150; 
  export let boxMargin = 5;
  export let variable = null;
  export let variable2 = null;

  export let garbageCollection = false;
  export let value2 = 42;
  export let translateXValue = 650;

  const pyObjectData = [ 
  {
    x: 0,
    y: boxHeight + boxMargin,
    content: 'Type'
  },  
  {
    x: boxWidth + boxMargin,
    y: boxHeight + boxMargin,
    content: type  
  },  
  {
    x: 0,
    y: (boxHeight + boxMargin)*2,
    content: 'Value'
  },  
  {
    x: boxWidth + boxMargin,
    y: (boxHeight + boxMargin) * 2,
    content: value  
  },  
  {
    x: 0,
    y: (boxHeight + boxMargin)*3,
    content: 'Count'
  },  
  {
    x: boxWidth + boxMargin,
    y: (boxHeight + boxMargin) * 3,
    content: garbageCollection ? count - 1 : count  
  },  
]

 const pyObjectData2 = [ 
  {
    x: translateXValue,
    y: boxHeight + boxMargin,
    content: 'Type'
  },  
  {
    x: boxWidth + boxMargin + translateXValue,
    y: boxHeight + boxMargin,
    content: type  
  },  
  {
    x: translateXValue,
    y: (boxHeight + boxMargin)*2,
    content: 'Value'
  },  
  {
    x: boxWidth + boxMargin + translateXValue,
    y: (boxHeight + boxMargin) * 2,
    content: value2  
  },  
  {
    x: translateXValue,
    y: (boxHeight + boxMargin)*3,
    content: 'Count'
  },  
  {
    x: boxWidth + boxMargin + translateXValue,
    y: (boxHeight + boxMargin) * 3,
    content: count  
  },  
]
</script>

<svg {width} {height}>
<g transform="translate(5, 5)">
  <g fill="none" stroke-width=2 stroke={garbageCollection ? 'var(--main-color-1)' : 'var(--text-color)'}>
   <rect x=0 y=0 width={boxWidth*2 + boxMargin} height={boxHeight}></rect>  
   <text stroke="none" 
        fill="var(--text-color)"
        x={boxWidth + boxMargin} 
        y={boxHeight/2} 
        dominant-baseline="middle" 
        text-anchor="middle">PyObject</text>
  </g>
  
  {#each pyObjectData as point} 
   <g transform="translate({point.x} {point.y})" fill="none" stroke-width=2 stroke={garbageCollection ? 'var(--main-color-1)' : 'var(--text-color)'}>
    <rect x=0 y=0 width={boxWidth} height={boxHeight}></rect>  
    <text stroke="none" 
         fill="var(--text-color)"
         x={boxWidth/2} 
         y={boxHeight/2} 
         dominant-baseline="middle" 
         text-anchor="middle">{point.content}</text>
   </g>
  {/each}

  {#if garbageCollection}
  <g fill="none" stroke-width=2 stroke="var(--text-color)">
   <rect x={translateXValue} y=0 width={boxWidth*2 + boxMargin} height={boxHeight}></rect>  
   <text stroke="none" 
        fill="var(--text-color)"
        x={boxWidth + boxMargin + translateXValue} 
        y={boxHeight/2} 
        dominant-baseline="middle" 
        text-anchor="middle">PyObject</text>
  </g>
  {/if}
  
  {#if garbageCollection}
  {#each pyObjectData2 as point} 
   <g transform="translate({point.x} {point.y})" fill="none" stroke-width=2 stroke="var(--text-color)">
    <rect x=0 y=0 width={boxWidth} height={boxHeight}></rect>  
    <text stroke="none" 
         fill="var(--text-color)"
         x={boxWidth/2} 
         y={boxHeight/2} 
         dominant-baseline="middle" 
         text-anchor="middle">{point.content}</text>
   </g>
  {/each}
  {/if}

  {#if variable}
    <rect fill="none" stroke="var(--text-color)" stroke-width=2 x=400 y={height/2 - boxHeight + boxMargin} width={boxWidth} height={boxHeight}> </rect>
    <text stroke="none" 
         fill="var(--text-color)"
         x={400 + boxWidth/2} 
         y={height/2 - boxHeight/2 + boxMargin} 
         dominant-baseline="middle" 
         text-anchor="middle">{variable}</text>
    {#if !garbageCollection}
    <path stroke="var(--text-color)" stroke-width=2 d="M {400} {height/2 - boxHeight/2 + boxMargin} h -90"  />    
    {:else if garbageCollection}
    <path stroke="var(--text-color)" stroke-width=2 d="M {400+boxWidth} {height/2 - boxHeight/2 + boxMargin} h +90"  />    
    {/if}
  {/if}
  {#if variable2}
    <rect fill="none" stroke="var(--text-color)" stroke-width=2 x=400 y={height/2 - boxHeight + boxMargin + 50} width={boxWidth} height={boxHeight}> </rect>
    <text stroke="none" 
         fill="var(--text-color)"
         x={400 + boxWidth/2} 
         y={height/2 - boxHeight/2 + boxMargin + 50} 
         dominant-baseline="middle" 
         text-anchor="middle">{variable2}</text>
    <path stroke="var(--text-color)" stroke-width=2 d="M {400} {height/2 - boxHeight/2 + boxMargin + 50} h -90"  />    
  {/if}
</g>
</svg>
