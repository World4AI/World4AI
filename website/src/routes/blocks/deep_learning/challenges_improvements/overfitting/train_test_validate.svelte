<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Plot from "$lib/Plot.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";

  let lossPathsData = [[], []];
  let lossTrain = 1;
  let lossValid = 1;
  for (let i = 0; i < 100; i++) {
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
</script>

<h1>Train, Test, Validate</h1>
<div class="separator" />
<Container>
  <p>
    We understand intuitively that overfitting leads to a model that does not
    generalize well to new unforseen data, but we would also like to have some
    tools that would allow us to measure the level of overfitting during the
    training process. It turns out that splitting the dataset into different
    buckets is essential to achieve the goal of measuring overfitting.
  </p>
  <div class="separator" />

  <h2>Data Splitting</h2>
  <p>
    All examples that we covered so far assumed that we have a single training
    dataset. In practice we split the dataset into preferably 3 sets. The
    training set, the validation set and the test set.
  </p>
  <p>
    The <Highlight>training</Highlight> set is the part that is actually used to
    train a neural network. This part of the data is used in the backpropagation
    algorithm, the other sets are never used to adjust the weights and biases of
    a neural network.
  </p>
  <p>
    The <Highlight>validation</Highlight> set is also used in the training process,
    but only during the performance measurement step. After each epoch we use the
    training and the validation sets to measure the loss. At first both losses decline,
    but after a while the validation loss (blue line) starts to increase again, the
    training loss (red line) on the other hand keeps decreasing. This is a strong
    indication that our model overfits to the training data and the larger the divergence,
    the larger the overfitting. The validation set simulates a situation, where the
    neural network encounters new data in the real world, so if we deploy the model
    with the final weights, the performance of the live model will most likely not
    correspond to our expectations. Therefore we might need to consider techniques
    to reduce overfitting. More on that in the next sections.
  </p>
  <Plot
    pathsData={lossPathsData}
    config={{
      minX: 0,
      maxX: 100,
      minY: 0,
      maxY: 1,
      xLabel: "Time",
      yLabel: "Loss",
      radius: 5,
      xTicks: [],
      yTicks: [],
      numTicks: 5,
      pathsColors: ["var(--main-color-1)", "var(--main-color-2)"],
    }}
  />
  <p>
    When we encounter overfitting by comparing the loss of the training and the
    validation dataset, we will most likely change several hyperparameters of
    the neural network in hope to reduce overfitting. It is not unlikely that we
    will continue doing that until we are satisfied with the performance. While
    we are not using the validation dataset directly in training, we are still
    observing the performance of the validation data and ajust accordingly, thus
    injecting some knowledge into the training of the weights and biases. At
    this point it is hard to argue that the validation dataset represents
    completely unforseen data. The <Highlight>test</Highlight> set on the other hand
    is not touched during the training process at all. The intention of having this
    additional dataset is to provide a method to test the performance of our model
    when it encounters truly never before seen data. We only use the data once. If
    we find out that we overfitted to the training and the validation dataset, we
    can not go back to tweak the parameters, because we would require a completely
    new test dataset, which we might not posess.
  </p>
  <p>
    While there are no hard rules when it comes to the proportions of your
    splits, there are some rules of thumb. A 10-10-80 split is for example
    relatively common.
  </p>
  <SvgContainer maxWidth="700px">
    <svg class="split" viewBox="0 0 500 40">
      <rect x="0" y="0" width="500" height="50" fill="var(--main-color-4)" />
      <line x1="50" y1="0" x2="50" y2="50" stroke="black" />
      <line x1="100" y1="0" x2="100" y2="50" stroke="black" />

      <text x="25" y="20">Test</text>
      <text x="75" y="20">Validate</text>
      <text x="300" y="20">Train</text>
    </svg>
  </SvgContainer>
  <p>
    In that case we want to keep the biggest chunk of our data for training and
    keep roughly 10 percent for validation and 10 percent for testing.
  </p>
  <div class="separator" />

  <h2>K-Fold Cross-Validation</h2>
  <div class="separator" />

  <h2>Stratification</h2>
  <div class="separator" />
</Container>

<style>
  .split {
    border: 2px solid black;
  }
  text {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 10px;
    vertical-align: middle;
    display: inline-block;
    font-weight: bold;
  }
</style>
