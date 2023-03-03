<script>
  export let x;
  export let y;
  export let r;
  export let color = "none";
  let className = "";
  export { className as class };
</script>

<circle cx={x} cy={y} {r} stroke="black" class={className} fill={color} />
