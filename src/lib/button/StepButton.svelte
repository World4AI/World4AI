<script>
  import { ArrowRightCircle } from "lucide-svelte";
  export let disabled = false;
</script>

<button
  class="outline-none border-none hover:scale-105"
  class:disabled
  {disabled}
  on:click
>
  <ArrowRightCircle
    aria-label="Take an animation step"
    class="fill-w4ai-yellow"
    size={40}
    strokeWidth={1}
  />
</button>

<style>
  .disabled {
    cursor: wait;
  }
</style>
