<script>
  export let width = 500;  
  export let height = 200;

  let inputSize = 2;
  let kernelSize = 3;
  let outputSize = 5;
  let stride = 2;

  let boxSize=20;
  let boxGap=2;
  let leftMargin = 20;

  let highlightedRow = 0;
  let highlightedCol = 0; 

  $: outputHighlightRowStart = highlightedRow * stride;
  $: outputHighlightRowEnd = outputHighlightRowStart + kernelSize;
  $: outputHighlightColStart = highlightedCol * stride;
  $: outputHighlightColEnd = outputHighlightColStart + kernelSize;
</script>

<svg viewBox="0 0 {width} {height}">
  <!--input -->   
  <text x={leftMargin} y={height/2 - 80}>Input</text>
  {#each Array(inputSize) as _, rowIdx}
    {#each Array(inputSize) as _, colIdx}
      <rect 
            class="clickable" 
            on:click={() => {
              highlightedRow = rowIdx;
              highlightedCol = colIdx;
              }}
            x={leftMargin + rowIdx*(boxSize+boxGap)} 
            y={height/2 - inputSize*(boxSize+boxGap)/2 + colIdx*(boxSize+boxGap)} 
            width={boxSize} 
            height={boxSize} 
            fill={rowIdx === highlightedRow && colIdx === highlightedCol ? "var(--main-color-1)" : "var(--main-color-2)"} 
            stroke="black"/>
    {/each}
  {/each}

  <!--kernel-->
  <text x={leftMargin + width/3} y={height/2 - 80}>Kernel</text>
  {#each Array(kernelSize) as _, rowIdx}
    {#each Array(kernelSize) as _, colIdx}
      <rect x={leftMargin + rowIdx*(boxSize+boxGap) + width/3} 
            y={height/2 - kernelSize*(boxSize+boxGap)/2 + colIdx*(boxSize+boxGap)} 
            width={boxSize} 
            height={boxSize} 
            fill="var(--main-color-3)" 
            stroke="black"/>
    {/each}
  {/each}

  <!--output-->
  <text x={leftMargin + width/3 * 2} y={height/2 - 80}>Output</text>
  {#each Array(outputSize) as _, rowIdx}
    {#each Array(outputSize) as _, colIdx}
      <rect x={leftMargin + rowIdx*(boxSize+boxGap) + width/3 * 2} 
            y={height/2 - outputSize*(boxSize+boxGap)/2 + colIdx*(boxSize+boxGap)} 
            width={boxSize} 
            height={boxSize} 
            fill={rowIdx >= outputHighlightRowStart 
                  && rowIdx < outputHighlightRowEnd
                  && colIdx >= outputHighlightColStart
                  && colIdx < outputHighlightColEnd
            ? "var(--main-color-1)" : "var(--main-color-4)"}
            stroke="black"/>
    {/each}
  {/each}
</svg>

<style>
  .clickable {
    cursor: pointer;
  }
</style>
