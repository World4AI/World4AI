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
    }
</style>