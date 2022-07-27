<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";

  import Plot from "$lib/plt/Plot.svelte";
  import Ticks from "$lib/plt/Ticks.svelte";
  import Path from "$lib/plt/Path.svelte";
  import XLabel from "$lib/plt/XLabel.svelte";
  import YLabel from "$lib/plt/YLabel.svelte";
  import Legend from "$lib/plt/Legend.svelte";

  let trainingHistory = [];
  let validationHistory = [];
  let lossTrain = 1;
  let lossValid = 1;
  for (let i = 0; i < 110; i++) {
    let x = i;
    let y = lossTrain;
    trainingHistory.push({ x, y });

    y = lossValid;
    validationHistory.push({ x, y });

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
  let divergencePath = [
    { x: 65, y: 0 },
    { x: 65, y: 1 },
  ];
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
    stopped the training you go back to the weights that showed the lowest
    validation loss. The assumption is, that those weights will exhibit best
    generalization capabilities.
  </p>

  <Plot
    width={500}
    height={300}
    maxWidth={700}
    domain={[0, 100]}
    range={[0, 1]}
  >
    <Path data={trainingHistory} color={"var(--main-color-1)"} />
    <Path data={validationHistory} />
    <Path data={divergencePath} />
    <Ticks
      xTicks={[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
      yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
      xOffset={-19}
      yOffset={18}
      fontSize={10}
    />
    <XLabel text={"Time"} fontSize={15} />
    <YLabel text={"Loss"} fontSize={15} />
    <Legend text="Validation Loss" coordinates={{ x: 75, y: 0.92 }} />
    <Legend
      text="Training Loss"
      coordinates={{ x: 75, y: 0.85 }}
      legendColor={"var(--main-color-1)"}
    />
  </Plot>
</Container>
