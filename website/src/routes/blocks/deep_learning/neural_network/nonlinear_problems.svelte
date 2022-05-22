<script>
  import Container from "$lib/Container.svelte";
  import Plot from "$lib/Plot.svelte";

  let pointsData = [[], []];
  let radius = [0.4, 0.25];
  let centerX = 0.5;
  let centerY = 0.5;
  for (let i = 0; i < radius.length; i++) {
    for (let point = 0; point < 200; point++) {
      let angle = 2 * Math.PI * Math.random();
      let r = radius[i];
      let x = r * Math.cos(angle) + centerX;
      let y = r * Math.sin(angle) + centerY;
      pointsData[i].push({ x, y });
    }
  }

  let numbers = 80;
  let heatmapData = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let classification;
      if (x + y > 1) {
        classification = 0;
      } else {
        classification = 1;
      }
      let coordinate = { x, y, class: classification };
      heatmapData.push(coordinate);
    }
  }

  let heatmapData2 = [];
  for (let i = 0; i < numbers; i++) {
    for (let j = 0; j < numbers; j++) {
      let x = i / numbers;
      let y = j / numbers;
      let classification;
      if ((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.1) {
        classification = 0;
      } else {
        classification = 1;
      }
      let coordinate = { x, y, class: classification };
      heatmapData2.push(coordinate);
    }
  }

  let config = {
    width: 500,
    height: 500,
    maxWidth: 600,
    minX: 0,
    maxX: 1,
    minY: 0,
    maxY: 1,
    xLabel: "Feature 1",
    yLabel: "Feature 2",
    padding: { top: 20, right: 40, bottom: 40, left: 60 },
    radius: 5,
    colors: ["var(--main-color-1)", "var(--main-color-2)", "var(--text-color)"],
    heatmapColors: ["var(--main-color-3)", "var(--main-color-4)"],
    xTicks: [],
    yTicks: [],
    numTicks: 5,
  };
</script>

<h1>Nonlinear Problems</h1>
<div class="separator" />

<Container>
  <p>What does nonlinearity mean?</p>
  <p>What does nonlinearity require?</p>
  <Plot {pointsData} {heatmapData} {config} />
  <Plot {pointsData} heatmapData={heatmapData2} {config} />
</Container>
