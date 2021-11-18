<script>
let width = 500;
let height = 200;
let stateSize = 50;
let padding = 20;
let markerWidth = 6;
let markerHeight = 3;
let probFontSize = 12;

export let transitions = [[0.2, 0.8], [0.5, 0.5]];
let yCenter = height / 2;
let xCenters = []; 
xCenters.push(stateSize / 2 + padding);
xCenters.push(width-stateSize / 2 - padding)
let activeState = 0;
</script>

<svg viewBox='0 0 {width} {height}'> 
    <defs>
    <marker id="arrowhead" markerwidth={markerWidth} markerheight={markerHeight} refX="0" refY={markerHeight/2} orient="auto" fill="var(--text-color)">
        <polygon points="0 0, {markerWidth} {markerHeight/2}, 0 {markerHeight}" />
    </marker>
    </defs>
    <!--draw states -->
    {#each transitions as transition, i}
      <rect stroke="black" fill="var(--main-color-1)" x={xCenters[i]-stateSize/2} y={yCenter-stateSize/2} width={stateSize} height={stateSize} />
      {#if i == activeState}
        <rect stroke="black" fill="var(--main-color-1)" opacity="0.2" x={xCenters[i] - stateSize / 2 - 5} y={yCenter - stateSize / 2 - 5} width={stateSize + 10} height={stateSize + 10} />
      {/if}
      <text stroke="none" fill="var(--text-color)" x={xCenters[i]} y={yCenter} dominant-baseline="mid+le" text-anchor="middle">
      S
      <tspan font-size="8" dx="-3" dy="8">{i}</tspan>
      </text>
    {/each}
    <!-- draw connections -->
    {#each transitions as transition, i}
      {#each transition as probability, k}
        {#if i==k}
          <path d="M {xCenters[i] + stateSize/2 * 0.8} {yCenter + stateSize/2} 
                V {yCenter + stateSize/2 + 30}
                H {xCenters[i] - stateSize/2 * 0.8} 
                V {yCenter + stateSize/2 + markerWidth}
                " 
                marker-end=url(#arrowhead)
                stroke="var(--text-color)"
                fill="transparent"
                />
          <text x={xCenters[i] - 10} y={yCenter + stateSize / 2 + 20} fill="var(--text-color)" font-size={probFontSize}>{probability}</text>
        {:else if k - i == 1}
          <path d="M {xCenters[i] + stateSize/2} 
                {yCenter - stateSize/2 * 0.8} 
                H {xCenters[k] - stateSize/2 - markerWidth}" 
                marker-end=url(#arrowhead)
                stroke="var(--text-color)" />
          <text x={xCenters[i] + stateSize} y={yCenter - 5} fill="var(--text-color)" font-size={probFontSize}>{probability}</text>
        {:else if i - k == 1}
          <path d="M {xCenters[i] - stateSize/2} 
                {yCenter + stateSize/2 * 0.8} 
                H {xCenters[k] + stateSize/2 + markerWidth}" 
                marker-end=url(#arrowhead)
                stroke="var(--text-color)" />
          <text x={xCenters[i] - stateSize} y={yCenter + 10} fill="var(--text-color)" font-size={probFontSize}>{probability}</text>
        {/if}
      {/each}
    {/each}
</svg>
