<script>
    import Math from '$lib/Math.svelte';
    import {draw} from 'svelte/transition';
    let timeStep = -1;
    let agentTimeStep = -1;
    let visible = false;
    let disabled = false;
    function handleClick(){
        timeStep++;
        visible = true;
        disabled = true;
        setTimeout(function(){
           disabled = false;
        }, 2500);
        setTimeout(function(){
           agentTimeStep = timeStep;
        }, 800);
        setTimeout(function(){
            visible = false;
        }, 2000);
    }
</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | MDP Sequential Interaction</title>
    <meta name="description" content="In reinforcement learning the Markov decison process can be seen as an interaction between the agent and the envrionment.">
</svelte:head>

<h1>MDP as Sequential Interaction</h1>
<p><em>How does interaction between the agent and the environment relate to the Markov decision process?</em></p>
<div class="separator"></div>
<p>In essence an MDP allows us to formalize the interaction loop between the agent and the environment, where the actions of the agent influence future states and rewards and the agent might have to decide to forego the current reward to get higher rewards in the future. The common assumption in reinforcement learning is the existence of an MDP at the core of each environment.</p>
<p class="info">A Markov decision process (MDP) is a formal description of a sequential decision problem with uncertainty.</p>
<p>The interaction is done sequentially, where the agent and the environment take turns to react to each other. Each iteration of actions, rewards and states happens in a period of time, called a time step, <Math latex={'t'} />. The time step is a discrete variable starting at 0 and increasing by 1 after each iteration. During the first time step the agent receives the initial state of the environment <Math latex={'S_0'} />  and reacts accordingly with the action <Math latex={'A_0'} />. The environment transitions into a new state <Math latex={'S_1'} />  and generates the reward <Math latex={'R_1'} />. The agent in turn reacts with the action <Math latex={'A_1'} /> and the interaction continues. The general notation of writing States, Actions and Rewards is <Math latex={'S_t, A_t, R_t'} />  where the subscript <Math latex={'t'} /> represents a particular time step.</p>

<div class="flex-center">
    <svg width="500" height="500" version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
     <defs>
      <marker id="TriangleOutL"stroke="var(--text-color)" fill="var(--text-color)" overflow="visible" orient="auto">
       <path transform="scale(.8)" d="m5.77 0-8.65 5v-10l8.65 5z" fill="context-stroke" fill-rule="evenodd" stroke="context-stroke" stroke-width="1pt"/>
      </marker>
     </defs>
     <g id="environment" fill="none" stroke="var(--text-color)">
      <rect x="20" y="182.96" width="120" height="120" opacity=".999" stroke-linejoin="round"/>
      <rect x="30" y="192.96" width="100" height="100" opacity=".999" stroke-linejoin="round"/>
      <path d="m40 192.96v100"/>
      <path d="m50 192.96v100"/>
      <path d="m60 192.96v100"/>
      <path d="m70 192.96v100"/>
      <path d="m80 192.96v100"/>
      <path d="m90 192.96v100"/>
      <path d="m100 192.96v100"/>
      <path d="m110 192.96v100"/>
      <path d="m120 192.96v100"/>
      <path d="m30 202.96h100"/>
      <path d="m30 212.96h100"/>
      <path d="m30 222.96h100"/>
      <path d="m30 232.96h100"/>
      <path d="m30 242.96h100"/>
      <path d="m30 252.96h100"/>
      <path d="m30 262.96h100"/>
      <path d="m30 272.96h100"/>
      <path d="m30 282.96h100"/>
     </g>
     <g id="agent" fill="none" stroke="var(--text-color)">
      <rect x="387.54" y="211.12" width="63.797" height="63.797" ry="0"/>
      <rect x="393.92" y="217.5" width="51.036" height="51.036" ry="0"/>
      <circle cx="419.44" cy="243.01" r="9.6947" fill-rule="evenodd" stroke-linejoin="round"/>
      <path d="m398 209.63v-13.333h-13.333v-13.333"/>
      <path d="m411.33 209.63v-26.667"/>
      <path d="m424.67 209.63v-26.667"/>
      <path d="m438 209.63v-13.333h13.333v-13.333"/>
      <path d="m451.33 222.96h13.333v-13.333h13.333"/>
      <path d="m451.33 236.3h26.667"/>
      <path d="m451.33 249.63h26.667"/>
      <path d="m451.33 262.96h13.333v13.333h13.333"/>
      <path d="m384.67 222.96h-13.333v-13.333h-13.333"/>
      <path d="m384.67 236.3h-26.667"/>
      <path d="m384.67 249.63h-26.667"/>
      <path d="m384.67 262.96h-13.333v13.333h-13.333"/>
      <path d="m398 276.3v13.333h-13.333v13.333"/>
      <path d="m411.33 276.3v26.667"/>
      <path d="m424.67 276.3v26.667"/>
      <path d="m438 276.3v13.333h13.333v13.333"/>
     </g>
     {#if timeStep >= 0} 
     <g stroke="var(--text-color)" fill="var(--text-color)" font-family="sans-serif">
      <text id="agent-text" x="367.98828" y="480.05859" font-size="40px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="367.98828" y="480.05859" font-family="sans-serif" font-size="40px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">Agent</tspan></text>
      <text id="environment-text" x="17.988281" y="40.058594" font-size="40px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="17.988281" y="40.058594" font-family="sans-serif" font-size="40px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">Environment</tspan></text>
      <g stroke-width=".76208">
       <text id="state-text" x="39.087296" y="130.36723" font-size="30.483px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="39.087296" y="130.36723" font-family="sans-serif" font-size="30.483px" font-weight="bold" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">S</tspan></text>
       <text id="state-underscore" x="60.958633" y="138.29228" font-size="18.667px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="60.958633" y="138.29228" font-family="sans-serif" font-size="18.667px" font-style="italic" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">{timeStep}</tspan></text>
       {#if timeStep >=1}
       <text id="reward-text" x="91.954735" y="130.98232" font-size="30.483px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="91.954735" y="130.98232" font-family="sans-serif" font-size="30.483px" font-weight="bold" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">R</tspan></text>
       <text id="reward-underscore" x="117.82607" y="138.90736" font-size="18.667px" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="117.82607" y="138.90736" font-family="sans-serif" font-size="18.667px" font-style="italic" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">{timeStep}</tspan></text>
       {/if}
      </g>
      <text id="timestep-text" x="210.60156" y="261.64844" font-size="32px" style="line-height:1.25" xml:space="preserve"><tspan x="210.60156" y="261.64844" font-family="sans-serif" font-size="32px" font-style="italic" font-weight="bold">t = </tspan></text>
      <text id="timestep-number" x="271.34631" y="261.64844" font-size="32px" style="line-height:1.25" xml:space="preserve"><tspan x="271.34631" y="261.64844" font-family="sans-serif" font-size="32px" font-style="italic" font-weight="bold">{timeStep}</tspan></text>
      {#if agentTimeStep >= 0}
      <text id="action-text" x="381.83002" y="357.41541" font-size="30.483px" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="381.83002" y="357.41541" font-family="sans-serif" font-size="30.483px" font-weight="bold" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">A</tspan></text>
      <text id="action-underscore" x="407.70132" y="365.34045" font-size="18.667px" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal;line-height:1.25" xml:space="preserve"><tspan x="407.70132" y="365.34045" font-family="sans-serif" font-size="18.667px" font-style="italic" stroke-width=".76208" style="font-variant-caps:normal;font-variant-east-asian:normal;font-variant-ligatures:normal;font-variant-numeric:normal">{agentTimeStep}</tspan></text>
      {/if}
     </g>
     {#if visible}
         <path in:draw="{{duration: 1000}}" id="path-up" d="m87 170v-90h340v90" fill="none" marker-end="url(#TriangleOutL)" stroke="var(--text-color)" stroke-width="1px"/>
         <path in:draw="{{duration: 1000, delay: 500}}" id="path-down" d="m440 321.18v88.818h-340v-88.818" fill="none" marker-end="url(#TriangleOutL)" stroke="var(--text-color)" stroke-width="1px"/>
     {/if}
    {/if}
    </svg>
</div>
<div class="flex-center">
    <button disabled={disabled} class:disabled="{disabled}" on:click|preventDefault={handleClick}>Step</button>
</div>
<p>The interactive example above shows how each timestep triggers a new state, a new reward and eventually a new action.</p>
<div class="separator"></div>

<style>
    button {
     background-color : var(--background-color);
     border: 1px solid var(--text-color);
     width: 20%;
     padding: 10px 5px;
     font-size: 20px;
     color: var(--text-color);
     cursor: pointer;
     text-transform: uppercase;

    }

    .disabled {
        color: var(--main-color-1);
        border-color: var(--main-color);
        cursor: wait;
    }
</style>
