<script>
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte";
  import { tweened } from "svelte/motion";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import XLabel from "$lib/plt/XLabel.svelte"; 
  import YLabel from "$lib/plt/YLabel.svelte"; 
  import Circle from "$lib/plt/Circle.svelte"; 
  import Path from '$lib/plt/Path.svelte';
  import Text from "$lib/plt/Text.svelte";


  export let animated = true;
  export let inputData = [];
  export let matrix = [];
  export let vector = [];
  export let activation = "identity";
  export let showText = true;

   
  let data = JSON.parse(JSON.stringify(inputData));
  //disable or enable play button
  let disabled = false;
  // indicates if transformation was already conducted
  let transformed = false;
  let store = tweened(inputData, { duration: 1000 });

  function reset() {
    data = JSON.parse(JSON.stringify(inputData));
    return store.set(inputData);
  }

  function linearTransform() {
    transformed = [];
    data.forEach((dataPoint) => {
      let x = dataPoint.x * matrix[0][0] + dataPoint.y * matrix[1][0];
      let y = dataPoint.x * matrix[0][1] + dataPoint.y * matrix[1][1];
      transformed.push({ x, y });
    });
    data = transformed;
    return store.set(data);
  }

  function translate() {
    transformed = [];
    data.forEach((dataPoint) => {
      let x = dataPoint.x + vector[0];
      let y = dataPoint.y + vector[1];
      transformed.push({ x, y });
    });
    data = transformed;
    return store.set(data);
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function relu(z){
    return Math.max(0, z);
  }

  function nonLinearTransform() {
    let transformed = [];
    let fn = activation === 'sigmoid' ? sigmoid : relu;
    data.forEach((dataPoint) => {
      let x = fn(dataPoint.x);
      let y = fn(dataPoint.y);
      transformed.push({ x, y });
    });
    data = transformed;
    return store.set(transformed);
  }

  async function tranform() {
    disabled = true;
    if (transformed) {
      await reset();
      transformed = false;
    } else {
      if (matrix.length > 0){
        await linearTransform();
      }
      if (vector.length > 0) {
        await translate();
      }
      if (activation != "identity") {
        await nonLinearTransform();
      }
      transformed = true;
    }
    disabled = false;
  }
</script>

{#if animated}
  <ButtonContainer>
    <StepButton {disabled} on:click={tranform} />
  </ButtonContainer>
{/if}
<Plot width={500} height={500} maxWidth={600} domain={[-3, 3]} range={[-3, 3]}>
  <Ticks xTicks={[-3, -2, -1, 0, 1, 2, 3]} 
         yTicks={[-3, -2, -1, 0, 1, 2, 3]} 
         xOffset={-15} 
         yOffset={15}/>
  <Path data={$store} isClosed={true} />
  <Circle data={$store} color="var(--main-color-1)" radius=5/>
  {#if showText}
    {#each $store as coordinates}
      <Text x={coordinates.x} y={coordinates.y+0.1} text={`x: ${coordinates.x.toFixed(2)} | y: ${coordinates.y.toFixed(2)}`}/>
    {/each}
  {/if}
  <XLabel text="Feature 1" fontSize={15} />
  <YLabel text="Feature 2" fontSize={15} />
</Plot>
