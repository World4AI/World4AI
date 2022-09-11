<script>
  import Slider from "$lib/Slider.svelte";
  import Latex from "$lib/Latex.svelte";

  //plotting library
  import Plot from "$lib/plt/Plot.svelte"; 
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import Path from "$lib/plt/Path.svelte"; 

  let width = 500;
  let height = 200;

  let bias = 0;
  let data = [];

  const xTicks = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
  const yTicks = [0, 1];

  function generateData(bias) {
    data = [];
    for (let i = -50; i <= 50; i++) {
      let x = i;
      let y;
      if (x < bias) {
        y = 0;
        data.push({ x, y });
      } else if (x > bias) {
        y = 1;
        data.push({ x, y });
      } else if (x === bias) {
        y = 0;
        data.push({ x, y });
        y = 1;
        data.push({ x, y });
      }
    }
  }

  $: generateData(bias);
</script>


<Plot {width} {height} maxWidth={700} domain={[-50, 50]} range={[0, 1]}>
  <Ticks xTicks={xTicks} yTicks={yTicks} xOffset={-12} yOffset={10}/>
  <Path data={data} /> 
</Plot>

<div class="flex-container">
  <div><Latex>\theta</Latex></div>
  <Slider bind:value={bias} min={-50} max={50} />
</div>

<style>
  .flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .flex-container div {
    width: 30px;
  }
</style>
