<script>
  export let width = 500;
  export let height = 200;
  // random or stratified
  export let type = "random";

  let numCategories = 10;
  let amount = 10;
  let boxWidth = height / numCategories;
  let gap = 2;

  let numbers = [];
  let randomCategories = [];
  let stratifiedCategories = [];

  for (let i = 0; i < numCategories; i++) {
    let category = [];
    for (let j = 0; j < amount; j++) {
      let value = i;
      let splitIndex;
      if (type === "random") {
        if (Math.random() >= 0.5) {
          splitIndex = 0;
        } else {
          splitIndex = 1;
        }
      } else if (type === "stratified") {
        if (j < amount / 2) {
          splitIndex = 0;
        } else {
          splitIndex = 1;
        }
      }
      category.push({ value, splitIndex });
    }
    numbers.push(category);
  }
</script>

<svg viewBox="0 0 {width} {height + gap * numCategories}">
  {#each numbers as category, catIdx}
    {#each category as number, numIdx}
      <rect
        x={numIdx * boxWidth + numIdx * gap + gap}
        y={catIdx * boxWidth + catIdx * gap + gap}
        fill="var(--main-color-3)"
        stroke="black"
        width={boxWidth}
        height={boxWidth}
      />

      <text
        x={numIdx * boxWidth + numIdx * gap + boxWidth / 2 + gap}
        y={catIdx * boxWidth + catIdx * gap + boxWidth / 2 + gap + 1}
        >{number.value}
      </text>
    {/each}
  {/each}
  {#each numbers as category, catIdx}
    {#each category as number, numIdx}
      <rect
        x={width - boxWidth - (numIdx * boxWidth + numIdx * gap + gap)}
        y={catIdx * boxWidth + catIdx * gap + gap}
        fill={number.splitIndex === 0
          ? "var(--main-color-1)"
          : "var(--main-color-2)"}
        stroke="black"
        width={boxWidth}
        height={boxWidth}
      />

      <text
        x={width - boxWidth / 2 - (numIdx * boxWidth + numIdx * gap + gap)}
        y={catIdx * boxWidth + catIdx * gap + boxWidth / 2 + gap + 1}
        >{number.value}
      </text>
    {/each}
  {/each}
</svg>

<style>
  text {
    dominant-baseline: middle;
    text-anchor: middle;
    font-size: 15px;
    vertical-align: middle;
    display: inline-block;
  }
</style>
