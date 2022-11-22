<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";

  let predictedX1 = 200;
  let predictedY1 = 30;
  let predictedX2 = 300;
  let predictedY2 = 130;

  let targetX1 = 250;
  let targetY1 = 50;
  let targetX2 = 350;
  let targetY2 = 150;

  // make sure not to overextend the variables
  $: if(predictedX1 >= predictedX2) {
    predictedX1 = predictedX2
  }

  $: if(predictedY1 >= predictedY2) {
    predictedY1 = predictedY2
  }

  $: intersectionX1 = Math.max(predictedX1, targetX1);
  $: intersectionY1 = Math.max(predictedY1, targetY1);
  $: intersectionX2 = Math.min(predictedX2, targetX2);
  $: intersectionY2 = Math.min(predictedY2, targetY2);

  $: intersection = Math.max((intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1), 0); 
  $: surfacePrediction =  (predictedX2 - predictedX1) * (predictedY2 - predictedY1); 
  let surfaceTarget =  (targetX2 - targetX1) * (targetY2 - targetY1); 

  $: union = surfacePrediction + surfaceTarget - intersection;
  $: iou = Math.max(intersection / union, 0);
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Intersection over Union</title>
  <meta
    name="description"
    content="Intersection over Union allows us to measure the quality of the predicted bounding box. The closer the IOU is to 1, the better the quality of the prediction." 
  />
</svelte:head>

<h1>Intersection over Union</h1>
<div class="separator" />
<Container>
  <p>The goal of object detection is to classify objects in the image (or video) and to draw a bounding box for each of those objects. That goal requires a good measure that tells the developer how for away is the predicted bounding box from the target bounding box. This measure is called the <Highlight>IOU</Highlight>, which is short for intersection over union.</p>
  <p>Below we see two bounding boxes. The yellow bounding box is the correct one that we will try to match as close as possible. The blue box is our prediction of the bounding box. There is an overlap between the prediction and the target bounding box. This is called <Highlight>intersection</Highlight> and is displayed in red color below. <Highlight>Union</Highlight> on the other hand is the area that encompasses all three colors. The IOU is simply the fraction of both values. </p>
  <Latex>{String.raw`IOU = \dfrac{intersection}{union}`}</Latex>
  <p>You can slide the values for the predicted bounding box and observe how the IOU changes. The IOU can fluctuate between 0 and 1. The closer the IOU is to 1 the closer the prediction is to the target bounding box.</p>

  <SvgContainer maxWidth="800px">
    <svg viewBox="0 0 500 200">
      <!-- target bbox -->
      <rect 
          x={targetX1} 
          y={targetY1} 
          width={targetX2-targetX1} 
          height=100 
          stroke="black" 
          fill="var(--main-color-3)" />
      <!-- predicted bbox -->
      <rect 
          x={predictedX1} 
          y={predictedY1} 
          width={predictedX2-predictedX1} 
          height={predictedY2-predictedY1} 
          stroke="black" 
          fill="var(--main-color-4)" />
      <!-- intersection -->
      <rect 
          x={intersectionX1} 
          y={intersectionY1} 
          width={Math.max(intersectionX2-intersectionX1, 0)} 
          height={Math.max(intersectionY2-intersectionY1, 0)} 
          stroke="black" 
          fill="var(--main-color-1)" />
      <text x=10 y=20>Intersection: {intersection}</text>
      <text x=10 y=40>Union: {union}</text>
      <text x=10 y=60>IOU: {iou.toFixed(4)}</text>
    </svg>
  </SvgContainer>
  <div class="flex">
    <span>X Left: {predictedX1}</span><Slider bind:value={predictedX1} min={0} max={500} step={1} />
  </div>
  <div class="flex">
    <span>X Right: {predictedX2}</span><Slider bind:value={predictedX2} min={0} max={500} step={1} />
  </div>
  <div class="flex">
    <span>Y Left: {predictedY1}</span><Slider bind:value={predictedY1} min={0} max={200} step={1} />
  </div>
  <div class="flex">
    <span>Y Right: {predictedY2}</span><Slider bind:value={predictedY2} min={0} max={200} step={1} />
  </div>
  <p>We will look at the exact calculations of the IOU once we start implementing our algorithm with PyTorch. Essentially it is just a matter of calculating the area of the boxes.</p>
  <div class="separator" />
</Container>

<style>
  svg {
    border: 1px solid black;
  }
  text {
    font-size: 12px;
  }
  .flex {
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
  }
  .flex span {
    width: 120px; 
    font-weight: 600;
  }
  @media(max-width: 768px) {
    text{
      font-size: 22px;
    }
  }
</style>


