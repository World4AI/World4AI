<script>
  export let x;
  export let y;
  export let radius = 3.5;

  let className = "";
  export { className as class };
</script>

<circle cx={x} cy={y} r={radius} class={className} fill="none" stroke="black" />
<circle cx={x} cy={y} r={1} fill="black" stroke="black" />
