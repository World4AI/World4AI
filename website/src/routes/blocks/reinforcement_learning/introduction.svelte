<script>
  import { onMount } from "svelte";

  onMount(() => {
    let interval = setInterval(() => {
      offset -= 20;
    }, 300);

    return () => clearInterval(interval);
  });
  let offset = 0;
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Introduction</title>
  <meta
    name="description"
    content="Reinforcement learning deals with encouraging good behaviour. The agent interacts with the environment and the environment provides feedback with the help of rewards."
  />
</svelte:head>

<h1>Introduction</h1>
<Question>What is reinforcement learning?</Question>
<div class="separator" />

<p>
  A reasonable way to think about reinforcement learning is to think about
  animals or even children whom you would like to teach good behaviour and
  discourage from bad behaviour. The way you would go about it would probably
  involve some reward when the animal (let’s say a dog) behaves the way you
  desire and punishment when it doesn’t. If you would like to teach the dog to
  sit down when you say the word <em>"sit"</em>, you could keep repeating the
  word until the dog sits down. When it does, give it a treat. If you keep doing
  that the dog should learn sooner or later to sit down when you give the
  command. The process would most likely take time and require a lot of
  patience, but animals are quite good at learning the right behaviour when it
  comes to acquiring food.
</p>
<svg version="1.1" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
  <g stroke="#000">
    <g id="maze" fill="var(--text-color)">
      <rect x="15" y="15" width="10" height="55" />
      <rect x="35" y="70" width="90" height="10" />
      <rect x="15" y="85" width="10" height="60" />
      <rect x="15" y="155" width="105" height="10" />
      <rect x="15" y="175" width="75" height="10" />
      <rect x="55" y="185" width="10" height="10" />
      <rect x="25" y="15" width="55" height="10" />
      <rect x="35" y="35" width="75" height="10" />
      <rect x="65" y="45" width="10" height="15" />
      <rect x="90" y="5" width="10" height="20" />
      <rect x="100" y="15" width="65" height="10" />
      <rect x="100" y="165" width="10" height="30" />
      <rect x="35" y="90" width="215" height="10" />
      <rect x="55" y="100" width="10" height="45" />
      <rect x="35" y="115" width="10" height="40" />
      <rect x="75" y="110" width="10" height="45" />
      <rect x="175" y="5" width="10" height="75" />
      <rect x="120" y="35" width="55" height="10" />
      <rect x="45" y="55" width="10" height="15" />
      <rect x="85" y="45" width="10" height="15" />
      <rect x="135" y="45" width="10" height="35" />
      <rect x="145" y="70" width="20" height="10" />
      <rect x="155" y="55" width="10" height="15" />
      <rect x="95" y="100" width="10" height="45" />
      <rect x="195" y="15" width="10" height="75" />
      <rect x="215" y="15" width="60" height="10" />
      <rect x="215" y="35" width="10" height="45" />
      <rect x="120" y="110" width="10" height="75" />
      <rect x="140" y="100" width="10" height="65" />
      <rect x="140" y="175" width="100" height="10" />
      <rect x="160" y="135" width="10" height="40" />
      <rect x="160" y="110" width="85" height="10" />
      <rect x="180" y="120" width="10" height="45" />
      <rect x="200" y="120" width="10" height="55" />
      <rect x="220" y="120" width="10" height="45" />
      <rect x="235" y="25" width="10" height="45" />
      <rect x="245" y="5" width="10" height="10" />
      <rect x="245" y="110" width="10" height="55" />
      <rect x="255" y="160" width="10" height="35" />
      <rect x="275" y="15" width="10" height="85" />
      <rect x="255" y="35" width="20" height="10" />
      <rect x="255" y="45" width="10" height="25" />
      <rect x="265" y="80" width="10" height="70" />
      <rect x="275" y="140" width="50" height="10" />
      <rect x="285" y="120" width="40" height="10" />
      <rect x="275" y="100" width="95" height="10" />
      <rect x="295" y="5" width="10" height="85" />
      <rect x="315" y="15" width="10" height="85" />
      <rect x="335" y="5" width="10" height="55" />
      <rect x="335" y="70" width="50" height="10" />
      <rect x="355" y="15" width="10" height="55" />
      <rect x="375" y="5" width="10" height="55" />
      <rect x="355" y="90" width="10" height="10" />
      <rect x="335" y="80" width="10" height="10" />
      <rect x="275" y="150" width="10" height="35" />
      <rect x="295" y="185" width="20" height="10" />
      <rect x="295" y="160" width="10" height="25" />
      <rect x="305" y="160" width="35" height="10" />
      <rect x="340" y="150" width="10" height="35" />
      <rect x="360" y="160" width="10" height="15" />
      <rect x="360" y="175" width="35" height="10" />
      <rect x="315" y="180" width="15" height="15" />
      <rect x="325" y="130" width="60" height="10" />
      <rect x="360" y="150" width="25" height="10" />
      <rect x="335" y="110" width="10" height="10" />
      <rect x="355" y="120" width="10" height="10" />
      <rect x="375" y="115" width="10" height="15" />
      <rect x="380" y="95" width="15" height="10" />
    </g>
    <g id="borders" fill="none" stroke="var(--text-color)">
      <path d="m5 5v190h380" stroke-width="1px" />
      <path d="m395 195v-190h-380" />
    </g>
    <path
      id="path-good"
      stroke="var(--main-color-2)"
      d="m10 10h75v20h30v35h15v20h60v-75h20v75h50v70h10v35h20v-35h45v-10h20v45h35v5"
      fill="none"
      stroke-dasharray="4,2,1,2"
      stroke-dashoffset={offset}
    />
    <path
      id="path-bad"
      stroke="var(--main-color-1)"
      d="m10 15v155h85v20h-25"
      fill="none"
      stroke-dasharray="4,2,1,2"
      stroke-dashoffset={offset}
    />
  </g>
</svg>
<p>
  We could also think about mice in a maze that get food once they arrive at the
  goal. The amount of food could depend on the time a mouse needs to complete
  the maze. If the mouse takes the blue path, the route is relatively straight
  forward and the mouse ends up at the goal without any delays. The red path on
  the other hand leads to a dead end, which forces the mouse to retrace the
  steps. Over time the mouse will learn the shourtest route which maximizes the
  amount of food.
</p>
<p>
  In reinforcement learning we similarly deal with agents which have to learn
  certain behaviour. The world (environment) that the agent interacts with
  provides reinforcement in form of numeric rewards which can be used to learn
  the desired behaviour.
</p>
<p>
  In later chapters we are going to introduce a more formal definition of
  reinforcement learning, but keep the above metaphor in mind. This intuition
  will guide you quite nicely in your journey through reinforcement learning.
</p>
<div class="separator" />
