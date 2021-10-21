<script>
    import Folder from '$lib/sidebar_components/Folder.svelte';

    let root = [
		{
			name: 'Intuition',
			links: [
				{
					name: 'Introduction',
					link: '/blocks/reinforcement_learning/intuition/introduction'
				},
				{
					name: 'Applications',
					link: '/blocks/reinforcement_learning/intuition/applications'
				},
				{
					name: 'Agent and Environment',
					link: '/blocks/reinforcement_learning/intuition/agent_and_environment'
				},
				{
					name: 'Properties of Reinforcement Learning',
					link: '/blocks/reinforcement_learning/intuition/properties'
				},
				{
					name: 'Exploration vs Exploitation',
					link: '/blocks/reinforcement_learning/intuition/exploration_vs_exploitation'
				},
				{
					name: 'Agent and Environment Components',
					link: '/blocks/reinforcement_learning/intuition/agent_and_environment_components'
				},
				{
					name: 'Reinforcement Learning Terminology',
					link: '/blocks/reinforcement_learning/intuition/terminology'
				},
			]
		},
		{
			name: 'Markov Decision Process',
			links: [
				{
					name: 'Introduction',
					link: '/blocks/reinforcement_learning/markov_decision_process/introduction'
				},
				{
					name: 'Definition',
					link: '/blocks/reinforcement_learning/markov_decision_process/definition'
				},
				{
					name: 'Solution',
					link: '/blocks/reinforcement_learning/markov_decision_process/solution'
				},			]
		}
	];

</script>

<div class="container">
	<div class="top"></div>
    <Folder name="ROOT" links={root}/>
</div>

<style>
    .container {
        background-color: var(--aside-color);
        min-height: 100vh;
		padding: var(--gap);
    }

	.top {
		height: 5vh;
	}
</style>