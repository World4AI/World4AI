<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  export let layers = [
    { width: 10, height: 10, channels: 3 },
    { width: 8, height: 8, channels: 8 },
    { width: 6, height: 6, channels: 16 },
    { width: 4, height: 4, channels: 32 },
    { width: 1, height: 1, channels: 64 },
  ];

  export let maxWidth = 1000;
  export let blockSize = 20;
  export let gap = 5;
  export let layerDistance = 300;

  let height = 400;
  let width = 1600;
</script>

<SvgContainer maxWidth={maxWidth + "px"}>
  <svg viewBox="0 0 {width} {height}">
    {#each layers as layer, layerIdx}
      <!-- move layers to the right -->
      <g transform="translate({layerIdx * layerDistance}, 0)">
        {#each Array(layer.channels) as _, channelIdx}
          <!-- move channels slightly to the bottom and right -->
          <g transform="translate({channelIdx * 5}, {channelIdx * 5})">
            {#each Array(layer.height) as _, rowIdx}
              {#each Array(layer.width) as _, pixelIdx}
                <rect
                  x={pixelIdx * (blockSize + gap)}
                  y={rowIdx * (blockSize + gap)}
                  width={blockSize}
                  height={blockSize}
                  fill={"white"}
                  stroke="black"
                />
              {/each}
            {/each}
          </g>
        {/each}
      </g>
    {/each}
  </svg>
</SvgContainer>
