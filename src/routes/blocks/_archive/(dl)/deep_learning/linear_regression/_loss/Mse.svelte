<script>
  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import XLabel from "$lib/plt/XLabel.svelte"; 
  import YLabel from "$lib/plt/YLabel.svelte"; 
  import Path from "$lib/plt/Path.svelte"; 
  import Legend from "$lib/plt/Legend.svelte"; 

  export let data = [
      { x: 5, y: 20 },
      { x: 10, y: 40 },
      { x: 35, y: 15 },
      { x: 45, y: 59 },
  ];

  export let w = 0;
  export let b = 0;

  let mse = 0;
  let regressionLine = [];
  let rectangles = [];

  function linesAndBoxes(w, b) {
    regressionLine = [];
    rectangles = [];
    mse = 0;

    //regression line
    let x1 = 0;
    let y1 = b + w * x1;
    let x2 = 60;
    let y2 = b + w * x2;
    regressionLine.push({x: x1, y: y1});
    regressionLine.push({x: x2, y: y2});

    data.forEach((point) => {
      let x1 = point.x;
      let x2 = point.x;
      let y1 = point.y;
      let y2 = b + w * x2;

      //sum squred error
      mse += (y1 - y2) ** 2;

      //rectangles
      let x = x1;
      let y = y1 > y2 ? y1 : y2;
      let size = Math.abs(y1 - y2);
      let rect = [
        { x, y },
        { x: x + size, y },
        { x: x + size, y: y - size },
        { x, y: y - size },
        { x, y },
      ];
      rectangles.push(rect);
    });
    mse = mse / data.length;
  }

  $: linesAndBoxes(w, b);
</script>

<Plot width={500} height={500} maxWidth={600} domain={[0, 60]} range={[0, 60]}>
  <Ticks xTicks={[0, 10, 20, 30, 40, 50, 60]} 
         yTicks={[0, 10, 20, 30, 40, 50, 60]} 
         xOffset={-15} 
         yOffset={15}/>
  <XLabel text="Feature" fontSize={15}/>
  <YLabel text="Target" fontSize={15}/>
  <Circle data={data} radius={5} />
  <Path data={regressionLine} />
  {#each rectangles as rectangle}
    <Path data={rectangle} />
  {/each}
  <Legend text="Weight {w.toFixed(2)}" coordinates={{x: 3, y:58}} />
  <Legend text="Bias {b.toFixed(2)}" coordinates={{x: 3, y:55}} />
  <Legend text="MSE {mse.toFixed(2)}" coordinates={{x: 3, y:52}} />
</Plot>
