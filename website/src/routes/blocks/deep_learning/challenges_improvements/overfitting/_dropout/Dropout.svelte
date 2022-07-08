<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import PlayButton from "$lib/PlayButton.svelte";

  export let width = 500;
  export let height = 250;

  export let nn = [
    [
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
    ],
    [
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
    ],
    [
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
    ],

    [
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
      { status: "active" },
    ],
    [{ status: "active" }],
  ];

  let border = 1;
  let rectSize = 20;
  let horizontalGap = (width - rectSize - border * 2) / (nn.length - 1);
  let verticalGap = 5;
  let p = 0.2;

  let offset = 0;
  //apply dropout
  function recalculateDropout() {
    nn.forEach((layer, layerIdx) => {
      if (layerIdx !== nn.length - 1) {
        layer.forEach((node, nodeIdx) => {
          if (Math.random() >= p) {
            nn[layerIdx][nodeIdx].status = "active";
          } else {
            nn[layerIdx][nodeIdx].status = "inactive";
          }
        });
      }
    });
    offset += 5;
  }

  let dropoutIntervalId = null;
  function dropoutHandler() {
    if (!dropoutIntervalId) {
      dropoutIntervalId = setInterval(recalculateDropout, 800);
    } else {
      clearInterval(dropoutIntervalId);
      dropoutIntervalId = null;
    }
  }
</script>

<PlayButton
  type={!dropoutIntervalId ? "play" : "pause"}
  on:click={dropoutHandler}
/>
<SvgContainer maxWidth="650px">
  <svg viewBox="0 0 {width} {height}">
    <!--connections-->
    {#each nn as layer, layerIdx}
      {#each layer as node, nodeIdx}
        {#if layerIdx !== nn.length - 1}
          {#each nn[layerIdx + 1] as nextNode, nextnodeIdx}
            <line
              x1={border + layerIdx * horizontalGap + rectSize}
              y1={(height -
                (layer.length - 1) * verticalGap -
                layer.length * rectSize) /
                2 +
                nodeIdx * (rectSize + verticalGap) +
                rectSize / 2}
              x2={border + (layerIdx + 1) * horizontalGap}
              y2={(height -
                (nn[layerIdx + 1].length - 1) * verticalGap -
                nn[layerIdx + 1].length * rectSize) /
                2 +
                nextnodeIdx * (rectSize + verticalGap) +
                rectSize / 2}
              stroke="var(--text-color)"
              stroke-width="0.8px"
              stroke-dasharray="5 2"
              stroke-opacity="0.4"
              stroke-dashoffset={offset}
            />
          {/each}
        {/if}
      {/each}
    {/each}
    <!--neurons-->
    {#each nn as layer, layerIdx}
      {#each layer as node, nodeIdx}
        <rect
          x={border + layerIdx * horizontalGap}
          y={(height -
            (layer.length - 1) * verticalGap -
            layer.length * rectSize) /
            2 +
            nodeIdx * (rectSize + verticalGap)}
          width={rectSize}
          height={rectSize}
          stroke="var(--text-color)"
          fill={node.status === "active"
            ? "var(--main-color-4)"
            : "var(--main-color-1)"}
        />
      {/each}
    {/each}
  </svg>
</SvgContainer>
