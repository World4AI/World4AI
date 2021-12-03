<script>
  import { fade } from 'svelte/transition';
  import Question from '$lib/Question.svelte';
</script>

<svelte:head>
    <title>World4AI | Programming | Data Types In Depth Intro</title>
    <meta name="description" content="It is essential to know what data types are best suited for what task.">
</svelte:head>

<h1>Python Data Types</h1>
<Question>Why do you need to know the basics of different data types?</Question>
<div class="separator"></div>
<div class="flex-center">
 <svg width="500" height="500" version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <g font-family="sans-serif" font-size="40px" fill="var(--text-color)">
   <text in:fade="{{duration: 1000, delay: 0}}" x="209.44733" y="389.10425" style="line-height:1.25" xml:space="preserve"><tspan x="209.44733" y="389.10425">"Hello World!"</tspan></text>
   <text in:fade="{{duration: 1000, delay: 500}}" x="44.925358" y="146.21457" style="line-height:1.25" xml:space="preserve"><tspan x="44.925358" y="146.21457">42</tspan></text>
   <text in:fade="{{duration: 1000, delay: 1000}}" x="40.636703" y="470.25629" style="line-height:1.25" xml:space="preserve"><tspan x="40.636703" y="470.25629">True</tspan></text>
   <text in:fade="{{duration: 1000, delay: 1500}}" x="14.070657" y="303.24518" style="line-height:1.25" xml:space="preserve"><tspan x="14.070657" y="303.24518">(1, 2, 3)</tspan></text>
   <text in:fade="{{duration: 1000, delay: 2000}}" x="322.04257" y="222.09315" style="line-height:1.25" xml:space="preserve"><tspan x="322.04257" y="222.09315">[1, 2, 3]</tspan></text>
   <text in:fade="{{duration: 1000, delay: 2500}}" x="201.53593" y="56.742237" style="line-height:1.25" xml:space="preserve"><tspan x="201.53593" y="56.742237">&#123;key: value&#125;</tspan></text>
  </g>
 </svg>
</div>
<div class="separator"></div>
<p>In light of the many functionalities that each data type posesses it is almost impossible to learn and cover every single aspect. But if there is something that we should take away from this chapter, it is the ability to decide which data type should be used in which circumstances?</p>
<div class="separator"></div>
