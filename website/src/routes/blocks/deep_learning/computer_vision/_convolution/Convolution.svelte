<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  export let imageWidth = 10;
  export let imageHeight = 10;
  export let maxWidth = 500;
  export let blockSize = 20;
  export let gap = 5;

  export let showOutput = false;
  export let showNumbers = false;

  // pixel values
  export let imageNumbers = [];
  export let kernelNumbers = [];
  let outputNumbers = [];

  export let filterLocation = { row: 0, column: 0 };
  export let kernel = 2;
  export let stride = 1;
  export let padding = 0;
  export let numFilters = 1;
  export let numChannels = 1;

  // if there is more than 1 channel or filter we need to adjust height
  let channelOffset = 10;
  let filterOffset = 5;

  let outputImageWidth = Math.floor(
    (imageWidth - kernel + 2 * padding) / stride + 1
  );
  let outputImageHeight = Math.floor(
    (imageHeight - kernel + 2 * padding) / stride + 1
  );

  // to center the output image
  let heightPadding =
    ((imageHeight - outputImageHeight) / 2) * (gap + blockSize) +
    padding * (blockSize + gap);

  let withAdjustment = 0;
  let imageDistance = 0;
  if (showOutput) {
    imageDistance = 50 * (padding + 1);
    withAdjustment =
      imageDistance +
      outputImageWidth * blockSize +
      (outputImageWidth - 1) * gap;
  }

  let height =
    imageHeight * blockSize +
    (imageHeight - 1) * gap +
    numChannels * channelOffset +
    padding * 2 * (blockSize + gap);
  let width =
    imageWidth * blockSize +
    (imageWidth - 1) * gap +
    withAdjustment +
    imageDistance +
    padding * 2 * (blockSize + gap);

  function slideWindow() {
    let row = filterLocation.row;
    let column = filterLocation.column;

    column += stride;
    if (column + kernel > imageWidth + padding * 2) {
      column = 0;
      row += stride;
    }

    if (row + kernel > imageHeight + padding * 2) {
      row = 0;
    }
    filterLocation = { row, column };
  }

  //fill numbers for image and kernel if not provided
  //and calculate output values
  if (showNumbers) {
    if (imageNumbers.length === 0) {
      for (let row = 0; row < imageHeight; row++) {
        let fullRow = [];
        for (let col = 0; col < imageWidth; col++) {
          let randNumber = Math.random();
          let num;
          if (randNumber < 0.33) {
            num = -1;
          } else if (randNumber < 0.66) {
            num = 0;
          } else {
            num = 1;
          }

          fullRow.push(num);
        }
        imageNumbers.push(fullRow);
      }
    }
    if (kernelNumbers.length === 0) {
      for (let row = 0; row < kernel; row++) {
        let fullRow = [];
        for (let col = 0; col < kernel; col++) {
          let num = Math.random() > 0.5 ? 1 : 0;
          fullRow.push(num);
        }
        kernelNumbers.push(fullRow);
      }
    }
    //calculate image output values
    imageNumbers.forEach((row, rowIdx) => {
      if (rowIdx + kernel <= imageHeight) {
        let fullRow = [];
        row.forEach((_, colIdx) => {
          if (colIdx + kernel <= imageWidth) {
            let value = 0;
            kernelNumbers.forEach((kernelRow, kernelRowIdx) => {
              kernelRow.forEach((kernelValue, kernelColIdx) => {
                value +=
                  imageNumbers[rowIdx + kernelRowIdx][colIdx + kernelColIdx] *
                  kernelValue;
              });
            });
            fullRow.push(value);
          }
        });
        outputNumbers.push(fullRow);
      }
    });
  }
</script>

{#if showOutput}
  <PlayButton f={slideWindow} delta={500} />
{/if}

<SvgContainer maxWidth={maxWidth + "px"}>
  <svg viewBox="0 0 {width} {height}">
    <!-- original image -->
    {#each Array(numChannels) as _, channelIdx}
      <g
        transform="translate({channelIdx * channelOffset}, {channelIdx *
          channelOffset})"
      >
        {#each Array(imageHeight + padding * 2) as _, rowIdx}
          {#each Array(imageWidth + padding * 2) as _, pixelIdx}
            <!-- padding -->
            <rect
              x={pixelIdx * (blockSize + gap)}
              y={rowIdx * (blockSize + gap)}
              width={blockSize}
              height={blockSize}
              fill={pixelIdx < filterLocation.column + kernel &&
              rowIdx < filterLocation.row + kernel &&
              pixelIdx >= filterLocation.column &&
              rowIdx >= filterLocation.row
                ? "var(--main-color-1)"
                : padding > 0 &&
                  (rowIdx < padding ||
                    rowIdx > imageHeight + padding - 1 ||
                    pixelIdx < padding ||
                    pixelIdx > imageWidth + padding - 1)
                ? "var(--main-color-3)"
                : "var(--main-color-4)"}
              stroke="black"
            />
          {/each}
        {/each}
      </g>
    {/each}

    <!-- output image -->
    {#if showOutput}
      {#each Array(numFilters) as _, filterIdx}
        <g
          transform="translate({filterIdx * filterOffset}, {filterIdx *
            filterOffset})"
        >
          {#each Array(outputImageHeight) as _, rowIdx}
            {#each Array(outputImageWidth) as _, pixelIdx}
              <rect
                x={imageWidth * (blockSize + gap) +
                  imageDistance +
                  pixelIdx * (blockSize + gap)}
                y={heightPadding + rowIdx * (blockSize + gap)}
                width={blockSize}
                height={blockSize}
                fill={filterLocation.row / stride == rowIdx &&
                filterLocation.column / stride == pixelIdx
                  ? "var(--main-color-1)"
                  : "var(--main-color-4)"}
                stroke="black"
              />
            {/each}
          {/each}
        </g>
      {/each}
    {/if}

    {#if showNumbers}
      <!-- image numbers -->
      {#each imageNumbers as row, rowIdx}
        {#each row as num, colIdx}
          <text
            fill="black"
            x={colIdx * (blockSize + gap) + blockSize / 2}
            y={rowIdx * (blockSize + gap) + blockSize / 2}>{num}</text
          >
        {/each}
      {/each}
      <!-- kernel numbers -->
      {#each kernelNumbers as row, rowIdx}
        {#each row as num, colIdx}
          <text
            class="kernel-number"
            x={(filterLocation.column + colIdx) * (blockSize + gap) +
              blockSize -
              3}
            y={(filterLocation.row + rowIdx) * (blockSize + gap) +
              blockSize -
              4}
            fill="var(--main-color-3)">{num}</text
          >
        {/each}
      {/each}
      <!-- output numbers -->

      {#if showOutput}
        {#each outputNumbers as row, rowIdx}
          {#each row as num, pixelIdx}
            <text
              x={imageWidth * (blockSize + gap) +
                imageDistance +
                pixelIdx * (blockSize + gap) +
                blockSize / 2}
              y={heightPadding + rowIdx * (blockSize + gap) + blockSize / 2}
              fill="black">{num}</text
            >
          {/each}
        {/each}
      {/if}
    {/if}
  </svg>
</SvgContainer>

<style>
  text {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 14px;
  }

  .kernel-number {
    font-size: 10px;
    font-weight: bold;
  }
</style>
