<script>
    import {onMount} from 'svelte';
    import Latex from '$lib/Math.svelte';
    import Mdp from '$lib/reinforcement_learning/markov_decision_process/Mdp.svelte';
        
    let width = 500;
    let height = 100;
    let radius = 12;
    let distance = 30;
    let bernouliList = [];
    let numberOfTosses = width / distance + 1;

    let tossProb = 0.5; 
    
    let bernouliProcess = () => {
        if(Math.random() < tossProb) {
            return 0;    
        }
        else {
            return 1
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
               x = width + radius + distance - 12; 
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
        <text class="text-style" x="-6.5" y="6.5" fill="black">{coin.type === 0 ? 'H' : 'T'}</text>
    </g>
    {/each}
</svg>
<p>In the animation above we see the result of an unlimited number of random coin tosses, where each toss results either in an <strong>H</strong>, the head of a coin, or a <strong>T</strong>, the tail of a coin. In this example we deal with a fair coin, that means that the probability of drawing eather heads or tails is always exaclty 50%. What we have described is the so called Bernoulli process. The process evolves randomly by continuosly flipping an imaginary fair coin.</p>
<svg viewBox="0 0 {width} {height}">
    {#each bernouliList as coin}
    <g transform="translate({coin.x}, {coin.y})">
        <circle cx="0" cy="0" r="{radius}" stroke="black" fill="var(--main-color-1)" />
        <text class="text-style" x="-6.5" y="6.5" fill="black">{coin.type}</text>
    </g>
    {/each}
</svg>
<p>For our purposes it is more convenient to represent heads and tails as a 0's and 1's. We can look at these numbers as states in an enviroment, where the environment transitions from one state into an another with a 50% probability and stays in the same state with 50% probability.</p>
<p>For the Bernoulli process we can state the following: <Latex latex={'Pr(S_{t+1} \\mid S_t) = Pr(S_{t+1})'} />. <Latex latex={'Pr(S_{t+1})'} /> is the probability that a certain state will be tossed, in the above case <Latex latex={'Pr(S_{t+1}=0) = 0.5'} /> and <Latex latex={'Pr(S_{t+1}=1)=0.5'} />. <Latex latex={'Pr(S_{t+1} \\mid S_t)'} />  , reads as <Latex latex={'S_{t+t}'} /> given <Latex latex={'S_t'} />, depicts a conditional probability, where the probability of being in the new state <Latex latex={'S_{t+1}'} />  depends on the current state <Latex latex={'S_t'} />. For example <Latex latex={'Pr(S_{t+1}=0|S_t=1)'} />  shows the probability of a coin toss having a value of HEADS when the previous toss had a value of TAILS. When you consider a coin toss, then the new occurrence of either heads or tails does not depend on the previous specific value of the toss. The events are independent. <Latex latex={'Pr(S_{t+1} \\mid S_t) = Pr(S_t)'} />  means that knowing the last value of a coin toss does not give us any more knowledge regarding the future toss and therefore <Latex latex={'Pr(S_{t+1} \\mid S_t) = Pr(S_t) = 0.5'} />.</p>
<p>If we used the Bernoulli process in reinforcement learning, our agent would drift from one state into the next state, without any agency and any rewards to guide his actions.</p>

<h2>Markov Chain</h2>
<p class='info'>A Markov chain is a stochastic process that has the Markov property.</p>
<Mdp />
<p>The above animation depicts a simple, two state, Markov chain. Unlike in the case of a Bernoulli process the next state is not independent of the current state. The probability of the next state depends on the current state. If the environment is in the 0's state the probability to remain is 20% and the probability to transition into the 1's state is 80%. If the environment is in the 1's state on the other hand, there is a 50/50 chance of either staying in the same state or transitioning into the 0's state.</p>
<p>The Markov chain exhibits a so called Markov property. In simple words that means that the probability of transitioning into the next state is only dependent on the current state. The whole history of past states is irrelevant. This property is sometimes called <strong>memorylessness</strong>.</p>
<p>Mathematically the Markov property can be depicted as follows.</p>
<Latex latex={'Pr[S_{t+1} \\mid S_t] = Pr[S_{t+1} \\mid S_1, .... , S_t]'} />
<h2>Markov Reward Process</h2>
<h2>Markov Decision Process</h2>

<style>
    .text-style {
        font: 20px sans-serif;
    }
</style>
