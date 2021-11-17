<script>
    import {onMount} from 'svelte';
    let width = 500;
    let height = 100;
    let radius = 12;
    let distance = 30;
    let bernouliList = [];
    let numberOfTosses = width / distance + 1;
    
    let bernouliProcess = () => {
        if(Math.random() < 0.5) {
            return 'H';    
        }
        else {
            return 'T';
        }
    };

    for (let i = 0; i <= numberOfTosses; i++){
        let toss = bernouliProcess();
        let coin = {type: toss, x: width + radius + distance*i, y: height / 2 }
        bernouliList.push(coin);
    }

    onMount(() => {
        let interval = setInterval(() => {
        bernouliList = bernouliList.map((coin) => {
            let x = coin.x -=1;
            let type = coin.type;
            if (x <= - radius) {
               type = bernouliProcess();
               x = width + radius + distance; 
            }
            return {...coin, x, type};    
        })
        }, 10);

        return () => clearInterval(interval);
    })
</script>
<h1>MDP as Stochastic Process</h1>
<p><em>What is a stochastic process and what is a Markov property?</em></p>
<div class="separator"></div>
<p>The name Markov decison process was not named out of thin air. It is categorized by three distinct parts. It is a stochastic <strong>process</strong>, it has the <strong>Markov</strong> property and it provides an interaction mechanism for the agent to make <strong>decisions</strong>.</p>
<h2>Stochastic Process</h2>
<p class="info">A stochastic process is defined as a sequence of random variables.</p>
<svg viewBox="0 0 {width} {height}">
    {#each bernouliList as coin}
    <g transform="translate({coin.x}, {coin.y})">
        <circle cx="0" cy="0" r="{radius}" stroke="black" fill="var(--main-color-1)" />
        <text class="text-style" x="-6.5" y="6.5" fill="black">{coin.type}</text>
    </g>
    {/each}
</svg>
<h2>Markov Chain</h2>
<h2>Markov Reward Process</h2>
<h2>Markov Decision Process</h2>

<svg viewBox="0 0 {width} {height}">
    {#each bernouliList as coin}
    <g transform="translate({coin.x}, {coin.y})">
        <circle cx="0" cy="0" r="{radius}" stroke="black" fill="var(--main-color-1)" />
        <text class="text-style" x="-6.5" y="6.5" fill="black">{coin.type}</text>
    </g>
    {/each}
</svg>

<style>
    .text-style {
        font: 20px sans-serif;
    }
</style>
