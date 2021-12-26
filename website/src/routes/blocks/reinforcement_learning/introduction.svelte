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
  <g fill="none" stroke="#000" stroke-linecap="square">
    <g id="lines" stroke-width="0.2px" stroke="var(--text-color)">
      <path d="m0 10v190h390" />
      <path d="m400 190v-190h-390" />
      <g>
        <path d="m15 70v-55" />
        <path d="m15 15h65" />
        <path d="m90 0v25" />
        <path d="m90 25h75" />
        <path d="m15 85v60" />
        <path d="m60 200v-25" />
        <path d="m15 175h75" />
        <path d="m35 35h75" />
        <path d="m70 35v25" />
        <path d="m90 35v25" />
        <path d="m50 55v25" />
        <path d="m35 80h90" />
        <path d="m185 0v80" />
        <path d="m185 40h-65" />
        <path d="m140 40v40" />
        <path d="m140 80h25" />
        <path d="m165 80v-30" />
        <path d="m35 95h215" />
        <path d="m200 95v-80" />
        <path d="m15 160h115" />
        <path d="m130 110v75" />
        <path d="m105 160v40" />
        <path d="m100 95v55" />
        <path d="m80 160v-55" />
        <path d="m60 95v55" />
        <path d="m40 160v-55" />
        <path d="m145 95v70" />
        <path d="m140 185h100" />
        <path d="m160 110h95v50h10v40" />
        <path d="m205 185v-75" />
        <path d="m225 110v65" />
        <path d="m185 110v65" />
        <path d="m165 185v-65" />
        <path d="m220 95v-60" />
        <path d="m215 15h70v85h85" />
        <path d="m250 15v-15" />
        <path d="m240 15v60" />
        <path d="m255 70v-35h30" />
        <path d="m300 85v-85" />
        <path d="m315 15v85" />
        <path d="m335 0v60" />
        <path d="m355 15v55" />
        <path d="m335 70h50" />
        <path d="m375 0v60" />
        <path d="m335 70v20" />
        <path d="m400 95h-20" />
        <path d="m355 100v-10" />
        <path d="m400 185h-35v-35h20" />
        <path d="m385 115v25h-60v-10h-40v-10" />
        <path d="m360 140v-20" />
        <path d="m340 115v25" />
        <path d="m265 80v70h60" />
        <path d="m280 150v35" />
        <path d="m295 200v-40h45v-10" />
        <path d="m340 170h-35v30" />
        <path d="m350 160v40" />
      </g>
    </g>
    <path
      id="path-bad"
      stroke="var(--main-color-1)"
      d="m5 10v160h90v10h-30v15h35v-30h-90v-15h15v-50h25v55h20v-55h20v55h25v-55h20v90h-10v-25h-15v30h30v-25h15v-70h105v55h15v40h10v-40h45v-10h25v50h40"
      stroke-dasharray="4, 2, 1, 2"
      stroke-dashoffset={offset}
    />
    <path
      id="path-good"
      stroke="var(--main-color-2)"
      d="m10 10h75v20h30v40h15v15h65v-75h15v15h20v60h30v-30h15v55h120v35h-35v45h35"
      stroke-dasharray="4, 2, 1, 2"
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
