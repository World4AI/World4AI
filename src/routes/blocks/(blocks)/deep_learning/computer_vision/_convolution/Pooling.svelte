<script>
  import SvgContainer from "$lib/SvgContainer.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  export let imageWidth = 4;
  export let imageHeight = 4;
  export let maxWidth = 500;
  export let blockSize = 20;
  export let gap = 5;

  // pixel values
  export let imageNumbers = [];
  let outputNumbers = [];

  export let filterLocation = { row: 0, column: 0 };
  let kernel = 2;
  let stride = kernel;

  let outputImageWidth = Math.floor((imageWidth - kernel) / stride + 1);
  let outputImageHeight = Math.floor((imageHeight - kernel) / stride + 1);

  // to center the output image
  let heightPadding =
    ((imageHeight - outputImageHeight) / 2) * (gap + blockSize);

  const imageDistance = 50;
  const withAdjustment =
    imageDistance + outputImageWidth * blockSize + (outputImageWidth - 1) * gap;

  let height = imageHeight * blockSize + (imageHeight - 1) * gap;
  let width =
    imageWidth * blockSize +
    (imageWidth - 1) * gap +
    withAdjustment +
    imageDistance;

  function slideWindow() {
    let row = filterLocation.row;
    let column = filterLocation.column;

    column += stride;
    if (column + kernel > imageWidth) {
      column = 0;
      row += stride;
    }

    if (row + kernel > imageHeight) {
      row = 0;
    }
    filterLocation = { row, column };
  }

  //calculate image output values
  imageNumbers.forEach((row, rowIdx) => {
    if (rowIdx % stride === 0) {
      let fullRow = [];
      row.forEach((_, colIdx) => {
        if (colIdx % stride === 0) {
          let maxValue = 0;
          for (let i = 0; i < kernel; i++) {
            for (let k = 0; k < kernel; k++) {
              let num = imageNumbers[rowIdx + i][colIdx + k];
              if (num > maxValue) {
                maxValue = num;
              }
            }
          }
          fullRow.push(maxValue);
        }
      });
      outputNumbers.push(fullRow);
    }
  });
</script>

<ButtonContainer>
  <PlayButton f={slideWindow} delta={500} />
</ButtonContainer>

<SvgContainer maxWidth={maxWidth + "px"}>
  <svg viewBox="0 0 {width + 2} {height + 2}">
    <!-- original image -->
    {#each Array(imageHeight) as _, rowIdx}
      {#each Array(imageWidth) as _, pixelIdx}
        <rect
          x={1 + pixelIdx * (blockSize + gap)}
          y={1 + rowIdx * (blockSize + gap)}
          width={blockSize}
          height={blockSize}
          class={`stroke ${
            pixelIdx < filterLocation.column + kernel &&
            rowIdx < filterLocation.row + kernel &&
            pixelIdx >= filterLocation.column &&
            rowIdx >= filterLocation.row
              ? "fill-lime-200"
              : "fill-slate-400"
          }`}
          stroke="black"
        />
      {/each}
    {/each}

    <!-- output image -->
    {#each Array(outputImageHeight) as _, rowIdx}
      {#each Array(outputImageWidth) as _, pixelIdx}
        <rect
          x={imageWidth * (blockSize + gap) +
            imageDistance +
            pixelIdx * (blockSize + gap)}
          y={heightPadding + rowIdx * (blockSize + gap)}
          width={blockSize}
          height={blockSize}
          class={`stroke-black ${
            filterLocation.row / stride == rowIdx &&
            filterLocation.column / stride == pixelIdx
              ? "fill-w4ai-red"
              : "fill-w4ai-lightblue"
          }`}
        />
      {/each}
    {/each}

    <!-- image numbers -->
    {#each imageNumbers as row, rowIdx}
      {#each row as num, colIdx}
        <text
          fill="black"
          x={1 + colIdx * (blockSize + gap) + blockSize / 2}
          y={1 + rowIdx * (blockSize + gap) + blockSize / 2}>{num}</text
        >
      {/each}
    {/each}

    <!-- output numbers -->
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
  </svg>
</SvgContainer>

<style>
  text {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 14px;
  }
</style>
