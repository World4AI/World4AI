<script>
  import Slider from "$lib/Slider.svelte";
  import Plot from "$lib/Plot.svelte";
  import Latex from "$lib/Latex.svelte";

  let degree = 10;
  let alpha = 1;
  let lambda = 0.001;
  let numEpochs = 500;

  let polynomialPoints = [];
  let sign = 1;
  let randomScale = 100;
  for (let i = 0; i <= 30; i++) {
    let x = i;
    if (i % 10 === 0) {
      sign *= -1;
    }
    let y = Math.random() * randomScale * sign;
    polynomialPoints.push({ x, y });
  }

  //numbers to scale features
  let featuresMinMax = [];
  for (let i = 0; i < degree; i++) {
    let min = Number.MAX_VALUE;
    let max = Number.MIN_VALUE;
    polynomialPoints.forEach((point) => {
      let value = point.x ** i;
      if (value > max) {
        max = value;
      } else if (value < min) {
        min = value;
      }
    });
    featuresMinMax.push({ min, max });
  }

  // scale the features and prepare the targets
  let features = [];
  let targets = [];
  polynomialPoints.forEach((point) => {
    targets.push(point.y);
    let sample = [];
    for (let i = 0; i < degree; i++) {
      if (i === 0) {
        sample.push(1);
      }
      //min max scalar
      else {
        let feature =
          (point.x ** i - featuresMinMax[i].min) /
          (featuresMinMax[i].max - featuresMinMax[i].min);
        sample.push(feature);
      }
    }
    features.push(sample);
  });

  //gradient descent
  function train(lambda = 0) {
    let weights = [];
    let gradients = [];
    for (let i = 0; i < degree; i++) {
      weights.push(Math.random());
      gradients.push(0);
    }
    for (let epoch = 0; epoch < numEpochs; epoch++) {
      // reset gradients
      gradients.forEach((gradient, idx) => {
        gradients[idx] = 0;
      });

      // create predictions
      let predictions = [];
      features.forEach((sample) => {
        let prediction = 0;
        weights.forEach((weight, idx) => {
          prediction += weight * sample[idx];
        });
        predictions.push(prediction);
      });

      // calculate gradients
      features.forEach((sample, sampleIdx) => {
        let dLdz = targets[sampleIdx] - predictions[sampleIdx];
        weights.forEach((weight, weightIdx) => {
          let dzdw = -(sample[weightIdx] ** weightIdx);
          let gradient = dLdz * dzdw;
          let l2 = weight * lambda;
          gradients[weightIdx] += gradient + l2;
        });
      });

      //divide gradients by number of samples
      gradients.forEach((gradient, weightIdx) => {
        gradients[weightIdx] = gradient / features.length;
      });
      // take step
      weights.forEach((weight, weightIdx) => {
        weights[weightIdx] = weight - alpha * gradients[weightIdx];
      });
    }
    return weights;
  }

  //create data to draw the fitted lines
  function createPath(weights) {
    let path = [];
    for (let i = 0; i < features.length; i += 1) {
      let x = i;
      let y = 0;
      weights.forEach((weight, idx) => {
        y += weight * features[i][idx];
      });
      path.push({ x, y });
    }
    return path;
  }

  let weights = train();
  let overfittedPath = createPath(weights);

  let lambdaPath = [];
  let l2Path = [];
  $: if (lambda) {
    weights = train(lambda);
    lambdaPath = createPath(weights);
    l2Path = [[...overfittedPath], [...lambdaPath]];
  }
</script>

<span class="yellow"><Latex>\lambda</Latex>: {lambda}</span>
<Plot
  pointsData={polynomialPoints}
  pathsData={l2Path}
  config={{
    width: 800,
    height: 300,
    maxWidth: 800,
    minX: 0,
    maxX: 30,
    minY: -100,
    maxY: 100,
    xLabel: "x",
    yLabel: "y",
    radius: 2,
    pathsColors: ["var(--main-color-1)", "var(--main-color-2)"],
  }}
/>
<Slider min="0.001" max="0.1" step="0.001" bind:value={lambda} />
