<script>
  import Button from "$lib/Button.svelte";
  import { draw } from "svelte/transition";
  import { tweened } from "svelte/motion";

  export let frozen = false;
  let stepsBeforeCopy = 3;
  let step = 0;

  let width = 500;
  let height = 200;
  let disabled = false;

  let cxStore = tweened(width / 2, { duration: 1400, delay: 1000 });
  let cyStore = tweened(height / 2, { duration: 1400, delay: 1000 });
  $: cx = $cxStore;
  $: cy = $cyStore;

  let targetCxStore = tweened(width / 2 - 20, { duration: 1400, delay: 1000 });
  let targetCyStore = tweened(height / 2 - 40, { duration: 1400, delay: 1000 });
  $: targetCx = $targetCxStore;
  $: targetCy = $targetCyStore;

  //needed to calculate the direction of the target
  let showTargetDirection = false;
  let targetDirectionX = 0;
  let targetDirectionY = 0;
  let maxDirection = 100;
  let minDirection = -100;
  let directionFactor;
  if (frozen) {
    directionFactor = 0.5;
  } else {
    directionFactor = 0.9;
  }

  //needed to calculate the direction of the estimate
  let showEstimateDirection = false;
  let estimateDirectionX = 0;
  let estimateDirectionY = 0;

  //screen variables
  const screenXStore = tweened(0, { duration: 500 });
  const screenYStore = tweened(0, { duration: 500 });
  $: screenX = $screenXStore;
  $: screenY = $screenYStore;

  function handleClick() {
    step += 1;
    //disable button
    disabled = true;
    //ESTIMATE
    //estimate direction is draw
    estimateDirectionX = (targetCx - cx) * directionFactor;
    estimateDirectionY = (targetCy - cy) * directionFactor;
    showEstimateDirection = true;

    //estimate moves
    cxStore.set(cx + estimateDirectionX);
    cyStore.set(cy + estimateDirectionY);
    setTimeout(function () {
      //estimate direction disappears
      showEstimateDirection = false;
    }, 1000);

    if (!frozen || step > stepsBeforeCopy) {
      step = 0;
      //TARGET
      //target direction is drawn
      targetDirectionX =
        Math.random() * (maxDirection - minDirection) + minDirection;
      targetDirectionY =
        Math.random() * (maxDirection - minDirection) + minDirection;
      showTargetDirection = true;

      //target moves
      targetCxStore.set(targetCx + targetDirectionX);
      targetCyStore.set(targetCy + targetDirectionY);
      setTimeout(function () {
        //target direction disappears
        showTargetDirection = false;
      }, 1000);
    }

    //SCREEN

    setTimeout(function () {
      //screen follows
      screenXStore.set(cx - width / 2);
      screenYStore.set(cy - height / 2);
      //enable button
      disabled = false;
    }, 3500);
  }
</script>

<svg viewBox="{screenX} {screenY} {width} {height}">
  <!-- Draw Target Direction -->
  {#if showTargetDirection}
    <line
      in:draw={{ duration: 1000 }}
      x1={targetCx}
      y1={targetCy}
      x2={targetCx + targetDirectionX}
      y2={targetCy + targetDirectionY}
      stroke="var(--text-color)"
    />
  {/if}
  <!-- Draw Target -->
  <circle
    fill="var(--main-color-1)"
    stroke="black"
    cx={targetCx}
    cy={targetCy}
    r="10"
  />
  <!-- Draw Estimate Direction -->
  {#if showEstimateDirection}
    <line
      in:draw={{ duration: 1000 }}
      x1={cx}
      y1={cy}
      x2={cx + estimateDirectionX}
      y2={cy + estimateDirectionY}
      stroke="var(--text-color)"
    />
  {/if}
  <!-- Draw Estimate -->
  <circle fill="var(--main-color-2)" stroke="black" {cx} {cy} r="10" />
</svg>
<div class="flex-center">
  <Button {disabled} on:click={handleClick} value={"Step"} />
</div>
