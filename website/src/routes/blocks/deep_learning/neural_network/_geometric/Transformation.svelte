<script>
  import Plot from "$lib/Plot.svelte";
  import PlayButton from "$lib/PlayButton.svelte";
  import { tweened } from "svelte/motion";

  //disable or enable play button
  let disabled = false;

  export let inputData = [
    [{ x: 1, y: 1 }],
    [{ x: 2, y: 0 }],
    [{ x: 1, y: -1 }],
    [{ x: 1, y: -2 }],
    [{ x: -1, y: -2 }],
    [{ x: -2, y: -1 }],
    [{ x: -2, y: 0 }],
    [{ x: -1, y: 1 }],
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
      let x = dataPoint[0].x * matrix[0][0] + dataPoint[0].y * matrix[1][0];
      let y = dataPoint[0].x * matrix[0][1] + dataPoint[0].y * matrix[1][1];
      linearData.push([{ x, y }]);
    });
    return store.set(linearData);
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function nonLinearTransform() {
    let transformedData = [];
    linearData.forEach((dataPoint) => {
      let x = sigmoid(dataPoint[0].x);
      let y = sigmoid(dataPoint[0].y);
      transformedData.push([{ x, y }]);
    });
    store.set(transformedData);
  }

  async function tranform() {
    disabled = true;
    await reset();
    await linearTransform();
    if (activation != "identity") {
      await nonLinearTransform();
    }
    disabled = false;
  }

  let pathsData = [];
  if (activation != "identity") {
    pathsData = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
  }
</script>

<PlayButton {disabled} on:click={tranform} />
<Plot
  pointsData={$store}
  {pathsData}
  config={{
    width: 500,
    height: 500,
    maxWidth: 600,
    minX: -3,
    maxX: 3,
    minY: -3,
    maxY: 3,
    xLabel: "Feature 1",
    yLabel: "Feature 2",
    padding: { top: 20, right: 40, bottom: 40, left: 60 },
    radius: 5,
    xTicks: [],
    yTicks: [],
    numTicks: 13,
    colors: [
      "var(--main-color-1)",
      "var(--main-color-2)",
      "var(--main-color-3)",
      "var(--main-color-4)",
      "green",
      "blue",
      "orange",
    ],
  }}
/>
