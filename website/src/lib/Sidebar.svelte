<script>
  import Folder from '$lib/sidebar_components/Folder.svelte';
	export let root;
</script>

<div class="container">
    <Folder name="ROOT" links={root}/>
</div>

<style>
    .container {
  		margin-top: 150px;
  		margin-bottom: 20px;
      background-color: var(--aside-color);
  		padding: var(--gap);
      max-height: 70vh;
      overflow: auto;
      scrollbar-width: thin;
      scrollbar-color: var(--aside-color);
    }
    /* style scrollbar */
    .container::-webkit-scrollbar {
        width: 5px;
    }
    .container::-webkit-scrollbar-thumb {
        background-color: #FFF;
    }
    .container::-webkit-scrollbar-track {
        background-color: #000;
    }
</style>
