<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/languages/python";
  import github from "svelte-highlight/styles/base16-github";

  export let code = "";
</script>

<svelte:head>
  {@html github}
</svelte:head>

<div class="border my-2 text-sm">
  <Highlight language={python} {code} />
</div>
