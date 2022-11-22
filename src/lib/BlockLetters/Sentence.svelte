<script>
  import Letter from "$lib/BlockLetters/Letter.svelte";
  export let sentence = " ";
  export let gap = 10;

  let width = sentence.length * 50 + (sentence.length - 1) * gap;
</script>

<svg viewBox="0 0 {width} 50">
  <g fill="var(--main-color-1)" stroke="black">
    {#each sentence as letter, idx}
      <Letter {letter} letterIdx={idx} translateX={(50 + gap) * idx} />
    {/each}
  </g>
</svg>
