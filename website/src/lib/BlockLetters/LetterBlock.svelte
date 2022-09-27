<script>
  import { tweened } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';

  export let x;
  export let y;
  export let width;
  export let height;

  export let blockId = 0;
  export let letterId = 0;

  let progress = tweened(-50, {
    duration: 400,
    delay: blockId*100 + letterId*500,
    easing: cubicOut,
  })

  progress.set(0);
</script>


<g transform="translate(0, {$progress})">
  <rect {x} {y} {width} {height} />
</g>
