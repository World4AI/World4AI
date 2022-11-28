<script>
  import { scaleLinear } from "d3-scale";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import Path from "$lib/plt/Path.svelte"; 
  let width = 500;
  let height = 200;

  const xTicks = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
  const yTicks = [0, 10, 20, 30, 40, 50];

  let data = [];
  function generateData() {
    data = [];
    for (let i = -50; i <= 50; i++) {
      let x = i;
      let y;
      if (x <= 0) {
        y = 0;
        data.push({ x, y });
      } else if (x > 0) {
        y = x;
        data.push({ x, y });
      }
    }
  }

  $: generateData();
</script>

<Plot {width} {height} domain={[-50, 50]} range={[0, 50]}>
  <Ticks xTicks={xTicks} yTicks={yTicks} xOffset={-12} yOffset={10}/>
  <Path data={data} /> 
</Plot>

