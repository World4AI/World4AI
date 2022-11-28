<script>
  import Plot from "$lib/plt/Plot.svelte"; 
  import Circle from "$lib/plt/Circle.svelte";
  import Ticks from "$lib/plt/Ticks.svelte"; 
  import XLabel from "$lib/plt/XLabel.svelte"; 
  import YLabel from "$lib/plt/YLabel.svelte"; 
  import Path from "$lib/plt/Path.svelte"; 
  import Text from "$lib/plt/Text.svelte"; 
  import Slider from "$lib/Slider.svelte";

  export let p = 0.5;
  let entropy = 0;
  $: if (p === 0 || p === 1) {
    entropy = 0;
  } else {
    entropy = -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  }
  $: pointsData = [{ x: p, y: entropy }];

  let pathsData = [];
  for (let i = 0.001; i < 1; i += 0.001) {
    let x = i;
    let y = -x * Math.log2(x) - (1 - x) * Math.log2(1 - x);
    let data = { x, y };
    pathsData.push(data);
  }
</script>

<Plot width={500} height={250} maxWidth={800} domain={[0, 1]} range={[0, 1]}>
 <Ticks xTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
        yTicks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} 
        xOffset={-15} 
        yOffset={15}/>
 <XLabel text="Probability" fontSize={15} />
 <YLabel text="Entropy" fontSize={15} />
 <Path data={pathsData} />
 <Circle data={pointsData} />
 <Text text="Probability Heads: {p.toFixed(3)}" x=0.2 y=0.6/>
 <Text text="Probability Tails: {(1-p).toFixed(3)}" x=0.2 y=0.5/>
 <Text text="Entropy: {entropy.toFixed(5)}" x=0.2 y=0.4/>
</Plot>
<Slider min={0} max={1} step={0.001} bind:value={p} />
