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

  //disable or enable play button
  let disabled = false;

  export let inputData = [
    { x: 1, y: 1 },
    { x: 2, y: 0 },
    { x: 1, y: -1 },
    { x: 1, y: -2 },
    { x: -1, y: -2 },
    { x: -2, y: -1 },
    { x: -2, y: 0 },
    { x: -1, y: 1 }
  ];

  export let matrix = [
    [1, 0],
    [0, 1],
  ];

  export let activation = "identity";

  let store = tweened(inputData, { duration: 1000 });

  function reset() {
    return store.set(inputData);
  }

  let linearData = [];
  function linearTransform() {
    linearData = [];
    inputData.forEach((dataPoint) => {
      let x = dataPoint.x * matrix[0][0] + dataPoint.y * matrix[1][0];
      let y = dataPoint.x * matrix[0][1] + dataPoint.y * matrix[1][1];
      linearData.push({ x, y });
    });
    return store.set(linearData);
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function nonLinearTransform() {
    let transformedData = [];
    linearData.forEach((dataPoint) => {
      let x = sigmoid(dataPoint.x);
      let y = sigmoid(dataPoint.y);
      transformedData.push({ x, y });
    });
    store.set(transformedData);
  }

  async function tranform() {
    disabled = true;
    await reset();
    await linearTransform();
    if (activation != "identity") {
      nonLinearTransform();
    }
    disabled = false;
  }
</script>

<ButtonContainer>
  <StepButton {disabled} on:click={tranform} />
</ButtonContainer>
<Plot width={500} height={500} maxWidth={600} domain={[-3, 3]} range={[-3, 3]}>
  <Ticks xTicks={[-3, -2, -1, 0, 1, 2, 3]} 
         yTicks={[-3, -2, -1, 0, 1, 2, 3]} 
         xOffset={-15} 
         yOffset={15}/>
  <Circle data={$store} color="var(--main-color-1)" radius=3/>
  <XLabel text="Feature 1" fontSize={15} />
  <YLabel text="Feature 2" fontSize={15} />
</Plot>
