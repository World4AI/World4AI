<script>
  import Container from "$lib/Container.svelte";  
  import SvgContainer from "$lib/SvgContainer.svelte";

  const width = 200;
  const height = 200;
  const boxSize = 5;
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Image Segmentation</title>
  <meta
    name="description"
    content="The task of image segmentation is to classify each single pixel into object categories and to create masks that can be used as color overlays on top of images."
  />
</svelte:head>

<h1>Image Segmentation</h1>
<div class="separator" />

<Container>
  <p>The task of image segmentation is to segment the image into different classes. This is done by categorizing each single pixel.</p>
  <p>Let's assume we have an 40x40 image with several objects that repersent either plus and minus signs.</p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 {width} {height}">
      {#each Array(width/boxSize) as _, rowIdx}
        {#each Array(height/boxSize) as _, colIdx}
          <rect x={colIdx*boxSize} y={rowIdx*boxSize} width={boxSize} height={boxSize} fill=none stroke="black" stroke-width="0.2" />
        {/each}
      {/each}

      <!-- plus signs -->
      <rect x={5*boxSize} y={5*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={6*boxSize} y={5*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={6*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={6*boxSize} y={4*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={7*boxSize} y={5*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />

      <rect x={15*boxSize} y={15*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={16*boxSize} y={15*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={16*boxSize} y={16*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={16*boxSize} y={14*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={17*boxSize} y={15*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />

      <rect x={26*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={27*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={27*boxSize} y={27*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={27*boxSize} y={25*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={28*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />

      <!-- minus signs -->
      <rect x={35*boxSize} y={2*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={36*boxSize} y={2*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={37*boxSize} y={2*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />

      <rect x={2*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={3*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={4*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />

      <rect x={20*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={21*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={22*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />

      <rect x={20*boxSize} y={36*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={21*boxSize} y={36*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
      <rect x={22*boxSize} y={36*boxSize} width={boxSize} height={boxSize} fill="black" stroke="black" stroke-width="0.2" />
    </svg>
  </SvgContainer>
  <p>Our segementation task would involve to categorize each of the 40x40 pixels either into plus, minus or the background category. This categorization can be depicted as a so called mask, where each of the categories is represented as a separate color.</p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 {width} {height}">
      {#each Array(width/boxSize) as _, rowIdx}
        {#each Array(height/boxSize) as _, colIdx}
          <rect x={colIdx*boxSize} y={rowIdx*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-2)" stroke="black" stroke-width="0.2" />
        {/each}
      {/each}

      <!-- plus signs -->
      <rect x={5*boxSize} y={5*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={6*boxSize} y={5*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={6*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={6*boxSize} y={4*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={7*boxSize} y={5*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />

      <rect x={15*boxSize} y={15*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={16*boxSize} y={15*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={16*boxSize} y={16*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={16*boxSize} y={14*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={17*boxSize} y={15*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />

      <rect x={26*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={27*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={27*boxSize} y={27*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={27*boxSize} y={25*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />
      <rect x={28*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-1)" stroke="black" stroke-width="0.2" />

      <!-- minus signs -->
      <rect x={35*boxSize} y={2*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={36*boxSize} y={2*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={37*boxSize} y={2*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />

      <rect x={2*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={3*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={4*boxSize} y={26*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />

      <rect x={20*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={21*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={22*boxSize} y={6*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />

      <rect x={20*boxSize} y={36*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={21*boxSize} y={36*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
      <rect x={22*boxSize} y={36*boxSize} width={boxSize} height={boxSize} fill="var(--main-color-3)" stroke="black" stroke-width="0.2" />
    </svg>
  </SvgContainer>
  <div class="separator" />
</Container>

<style>
  svg {
    border: 1px solid black;
  }
</style>

