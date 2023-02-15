<script>
  export let x;
  export let y;
  export let radius = 3.5;
  export let offset = 1;
  export let color = "none";
</script>

<circle cx={x} cy={y} r={radius} fill={color} stroke="black" />
<line
  x1={x}
  y1={y - radius + offset}
  x2={x}
  y2={y + radius - offset}
  stroke="black"
/>
<line
  x1={x + radius - offset}
  y1={y}
  x2={x - radius + offset}
  y2={y}
  stroke="black"
/>
