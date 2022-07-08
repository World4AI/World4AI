<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Plot from "$lib/Plot.svelte";

  let lossPathsData = [[], [], []];
  let lossTrain = 1;
  let lossValid = 1;
  for (let i = 0; i < 110; i++) {
    let x = i;
    let y = lossTrain;
    lossPathsData[0].push({ x, y });

    y = lossValid;
    lossPathsData[1].push({ x, y });

    lossTrain *= 0.94;
    if (i <= 30) {
      lossValid *= 0.95;
    } else if (i <= 40) {
      lossValid *= 0.96;
    } else if (i <= 50) {
      lossValid *= 0.97;
    } else if (i <= 60) {
      lossValid *= 0.98;
    } else if (i <= 65) {
      lossValid *= 1;
    } else if (i <= 70) {
      lossValid *= 1.01;
    } else {
      lossValid *= 1.03;
    }
  }
  lossPathsData[2].push({ x: 65, y: 0 }, { x: 65, y: 1 });
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Early Stopping</title>
  <meta
    name="description"
    content="Early stopping is a simple technique to reduce overfitting by stopping the trainig process when the validation loss starts growing."
  />
</svelte:head>

<h1>Early Stopping</h1>
<div class="separator" />

<Container>
  <p>
    A simple strategy to deal with overfitting is to interrupt training, once
    the validation loss has been increasing for a certain number of epochs. When
    the validation loss starts increasing, while the training loss keeps
    decreasing, it is reasonable to assume that the training process has entered
    the phase of overfitting. At that point we should not waste the time
    watching the divergence between the training and valuation loss increase.
    This strategy is called <Highlight>early stopping</Highlight>. After you
    stopped the training you usually go back to the weights that showed the
    lowest validation loss. The assumption is, that those weights will
    generalize the best to the new unforseen data.
  </p>

  <Plot
    pathsData={lossPathsData}
    config={{
      minX: 0,
      maxX: 110,
      minY: 0,
      maxY: 1,
      xLabel: "Epochs",
      yLabel: "Loss",
      radius: 5,
      xTicks: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
      yTicks: [],
      numTicks: 5,
      pathsColors: ["var(--main-color-1)", "var(--main-color-2)", "red"],
    }}
  />
</Container>
