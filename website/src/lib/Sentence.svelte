<script>
  import Letter from "./Letter.svelte";

  export let sentence = " ";
</script>

<svg viewBox="0 0 {sentence.length * 60 - 10} 50">
  {#each sentence as letter, idx}
    <Letter {letter} translateX={idx * 50 + idx * 10} />
  {/each}
</svg>
