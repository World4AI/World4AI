<script>
  export let letter = " ";
  export let translateX = 0;
  export let translateY = 0;
  export let letterIdx = 0;
  export let letterDelay = 1000;
  export let blockDelay = 100;

	import { tweened } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';


  let letterMap = {
    a: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 40.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 20.5, width: 49, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 19 },
      { x: 40.5, y: 10.5, width: 9, height: 9 },
      { x: 10.5, y: 0.5, width: 39, height: 9 },
    ],
    b: [
      { x: 0.5, y: 40.5, width: 39, height: 9 },
      { x: 40.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 10.5, width: 9, height: 29 },
      { x: 10.5, y: 20.5, width: 39, height: 9 },
      { x: 30.5, y: 10.5, width: 9, height: 9 },
      { x: 0.5, y: 0.5, width: 39, height: 9 },
    ],
    c: [
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 29 },
    ],
    d: [
      { x: 0.5, y: 20.5, width: 9, height: 19 },
      { x: 0.5, y: 40.5, width: 39, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 19 },
      { x: 40.5, y: 30.5, width: 9, height: 19 },
      { x: 10.5, y: 20.5, width: 39, height: 9 },
    ],
    e: [
      { x: 0.5, y: 20.5, width: 39, height: 9 },
      { x: 0.5, y: 30.5, width: 9, height: 9 },
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 9 },
    ],
    f: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 20.5, width: 49, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 9 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
    ],
    g: [
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 40.5, y: 30.5, width: 9, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 29 },
      { x: 20.5, y: 20.5, width: 29, height: 9 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
    ],
    h: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 40.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 20.5, width: 49, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 19 },
      { x: 40.5, y: 0.5, width: 9, height: 19 },
    ],
    i: [
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 20.5, y: 10.5, width: 9, height: 29 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
    ],
    j: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 10.5, y: 40.5, width: 39, height: 9 },
      { x: 40.5, y: 10.5, width: 9, height: 29 },
      { x: 30.5, y: 0.5, width: 19, height: 9 },
    ],
    k: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 30.5, y: 40.5, width: 19, height: 9 },
      { x: 20.5, y: 30.5, width: 19, height: 9 },
      { x: 0.5, y: 20.5, width: 29, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 19 },
      { x: 20.5, y: 10.5, width: 19, height: 9 },
      { x: 30.5, y: 0.5, width: 19, height: 9 },
    ],
    l: [
      { x: 10.5, y: 40.5, width: 39, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 49 },
    ],
    m: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 40.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 20.5, width: 24, height: 9 },
      { x: 25.5, y: 20.5, width: 24, height: 9 },
      { x: 0.5, y: 10.5, width: 19, height: 9 },
      { x: 30.5, y: 10.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 9 },
    ],
    n: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 40.5, y: 40.5, width: 9, height: 9 },
      { x: 30.5, y: 30.5, width: 19, height: 9 },
      { x: 40.5, y: 20.5, width: 9, height: 9 },
      { x: 0.5, y: 20.5, width: 29, height: 9 },
      { x: 0.5, y: 10.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 19 },
    ],
    o: [
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 39 },
      { x: 40.5, y: 10.5, width: 9, height: 29 },
      { x: 10.5, y: 0.5, width: 39, height: 9 },
    ],
    p: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 20.5, width: 49, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 19 },
      { x: 0.5, y: 0.5, width: 39, height: 9 },
    ],
    q: [
      { x: 0.5, y: 40.5, width: 39, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 29 },
      { x: 30.5, y: 30.5, width: 19, height: 9 },
      { x: 30.5, y: 0.5, width: 9, height: 29 },
      { x: 0.5, y: 0.5, width: 29, height: 9 },
    ],
    r: [
      { x: 0.5, y: 30.5, width: 9, height: 19 },
      { x: 30.5, y: 30.5, width: 9, height: 19 },
      { x: 0.5, y: 20.5, width: 49, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 19 },
      { x: 40.5, y: 10.5, width: 9, height: 9 },
      { x: 10.5, y: 0.5, width: 39, height: 9 },
    ],
    s: [
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 40.5, y: 30.5, width: 9, height: 9 },
      { x: 0.5, y: 20.5, width: 49, height: 9 },
      { x: 0.5, y: 10.5, width: 9, height: 9 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
    ],
    t: [
      { x: 20.5, y: 10.5, width: 9, height: 39 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
    ],
    u: [
      { x: 0.5, y: 0.5, width: 9, height: 39 },
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 39 },
    ],
    v: [
      { x: 20.5, y: 40.5, width: 9, height: 9 },
      { x: 10.5, y: 30.5, width: 29, height: 9 },
      { x: 0.5, y: 20.5, width: 19, height: 9 },
      { x: 30.5, y: 20.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 19 },
      { x: 40.5, y: 0.5, width: 9, height: 19 },
    ],
    w: [
      { x: 0.5, y: 40.5, width: 19, height: 9 },
      { x: 20.5, y: 30.5, width: 9, height: 19 },
      { x: 30.5, y: 40.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 39 },
      { x: 40.5, y: 0.5, width: 9, height: 39 },
    ],
    x: [
      { x: 0.5, y: 40.5, width: 9, height: 9 },
      { x: 40.5, y: 40.5, width: 9, height: 9 },
      { x: 0.5, y: 30.5, width: 19, height: 9 },
      { x: 30.5, y: 30.5, width: 19, height: 9 },
      { x: 10.5, y: 20.5, width: 29, height: 9 },
      { x: 0.5, y: 10.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 9 },
      { x: 30.5, y: 10.5, width: 19, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 9 },
    ],
    y: [
      { x: 20.5, y: 30.5, width: 9, height: 19 },
      { x: 10.5, y: 20.5, width: 29, height: 9 },
      { x: 0.5, y: 10.5, width: 19, height: 9 },
      { x: 30.5, y: 10.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 9, height: 9 },
      { x: 40.5, y: 0.5, width: 9, height: 9 },
    ],
    z: [
      { x: 0.5, y: 40.5, width: 49, height: 9 },
      { x: 0.5, y: 30.5, width: 19, height: 9 },
      { x: 10.5, y: 20.5, width: 29, height: 9 },
      { x: 30.5, y: 10.5, width: 19, height: 9 },
      { x: 0.5, y: 0.5, width: 49, height: 9 },
    ],
    " ": [],
  };

  const progressCollection = [];
  for (let i = 0; i < letterMap[letter].length; i++) {
    let progress = tweened(-10, {
      duration: 400,
      easing: cubicOut
    })
    progressCollection.push(progress);
  }
</script>

<g id="letter-{letter}" transform="translate({translateX}, {translateY})">
  {#each letterMap[letter] as rect, idx}
    <g transform="translate(0, {$progressCollection[idx]})">
      <rect x={rect.x} y={rect.y} width={rect.width} height={rect.height} />
    </g>
  {/each}
</g>
