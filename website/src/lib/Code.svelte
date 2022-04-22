<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/languages/python";
  import "svelte-highlight/styles/nord.css";

  export let code;
</script>

<Highlight language={python} {code} langtag />
