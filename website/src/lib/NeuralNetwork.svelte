<script>
  export let width = 400;
  export let height = 150;
  export let padding = 5;
  export let radius = 5;

  export let config = {
    parameters: {
      0: { layer: 0, type: "input", count: 4, annotation: "Input" },
      1: { layer: 1, type: "fc", count: 7, input: [0] },
      2: { layer: 2, type: "fc", count: 5, input: [1] },
      3: { layer: 2, type: "fc", count: 5, input: [1] },
      4: {
        layer: 3,
        type: "fc",
        count: 5,
        input: [2],
        color: "var(--main-color-2)",
      },
      5: {
        layer: 3,
        type: "fc",
        count: 5,
        input: [3],
        color: "var(--main-color-1)",
      },
      6: {
        layer: 4,
        type: "fc",
        count: 2,
        input: [4],
        color: "var(--main-color-2)",
      },
      7: {
        layer: 4,
        type: "fc",
        count: 1,
        input: [5],
        color: "var(--main-color-1)",
      },

      8: {
        layer: 5,
        input: [6, 7],
        type: "addition",
        count: 2,
        annotation: "Q(s, a)",
      },
    },
  };

  // this loop figures out how many divisions there are in individual layers and how many layers there are
  // divisions happen when there are several separate calculation streams in a layer (for example duelling network)
  let layers = {};
  let maxCount = 0;
  let length = 0;
  let prevLayer = -1;
  for (const key in config.parameters) {
    let layer = config.parameters[key].layer;
    if (layer != prevLayer) {
      if (prevLayer != -1) {
        layers[prevLayer]["maxCount"] = maxCount;
      }
      length++;
      prevLayer = layer;
      maxCount = config.parameters[key].count;
    } else {
      if (config.parameters[key] > maxCount) {
        maxCount = config.parameters[key];
      }
    }
    let count = config.parameters[key].count;
    if (layer in layers) {
      layers[layer]["count"] += count;
      layers[layer]["divisions"] += 1;
      config.parameters[key]["division"] = layers[layer]["divisions"];
    } else {
      layers[layer] = {};
      layers[layer]["count"] = count;
      layers[layer]["divisions"] = 1;
      config.parameters[key]["division"] = 1;
    }
  }

  config.length = length;

  // this loop calculates the coordinates of the neurons
  for (const key in config.parameters) {
    let neuralNetwork = [];
    let cx =
      ((width - radius * 2 - padding * 2) / (config.length - 1)) *
        config.parameters[key].layer +
      radius +
      padding;
    let count = config.parameters[key].count;

    let divisions = layers[config.parameters[key].layer].divisions;
    let division = config.parameters[key].division;

    let offset = (height / divisions) * (division - 1);
    let heightPiece = height / divisions / count;
    for (let neuron = 0; neuron < count; neuron++) {
      let cy = offset + heightPiece * (neuron + 1) - heightPiece / 2;
      let coordinate = { cx, cy };
      neuralNetwork.push(coordinate);
    }
    config.parameters[key]["coordinates"] = neuralNetwork;
  }
</script>

<svg viewBox="0 0 {width} {height}">
  {#each Object.keys(config.parameters) as key}
    {#each config.parameters[key].coordinates as node, nodeIdx}
      <!-- first check if the nodes have an input -->
      {#if config.parameters[key].input}
        <!-- draw connections -->
        {#each config.parameters[key].input as prevKey}
          {#if config.parameters[key].type === "fc"}
            {#each config.parameters[prevKey].coordinates as prevNode}
              <line
                x1={node.cx}
                y1={node.cy}
                x2={prevNode.cx}
                y2={prevNode.cy}
                stroke={config.parameters[key].color
                  ? config.parameters[key].color
                  : "var(--text-color)"}
                stroke-width="0.1px"
              />
            {/each}
          {:else if config.parameters[key].type === "addition"}
            {#each config.parameters[prevKey].coordinates as prevNode, prevNodeIdx}
              {#if nodeIdx === prevNodeIdx || config.parameters[prevKey]["count"] != layers[config.parameters[prevKey].layer]["maxCount"]}
                <line
                  x1={node.cx}
                  y1={node.cy}
                  x2={prevNode.cx}
                  y2={prevNode.cy}
                  stroke={config.parameters[key].color
                    ? config.parameters[key].color
                    : "var(--text-color)"}
                  stroke-width="0.1px"
                />
              {/if}
            {/each}
          {/if}
        {/each}
      {/if}

      <!-- draw neurons -->
      <circle
        cx={node.cx}
        cy={node.cy}
        r={radius}
        fill="var(--background-color)"
        stroke="var(--text-color)"
      />
      <!-- draw addition -->
      {#if config.parameters[key].type == "addition"}
        <line
          x1={node.cx - radius / 2}
          y1={node.cy}
          x2={node.cx + radius / 2}
          y2={node.cy}
          stroke="var(--text-color)"
        />
        <line
          x1={node.cx}
          y1={node.cy - radius / 2}
          x2={node.cx}
          y2={node.cy + radius / 2}
          stroke="var(--text-color)"
        />
      {/if}
    {/each}
  {/each}
</svg>
