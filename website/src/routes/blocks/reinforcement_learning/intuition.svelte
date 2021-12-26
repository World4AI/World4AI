<script>
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Intuition</title>
  <meta
    name="description"
    content="Reinforcement learning can be taught in a an intuitive way. World4AI teaches reinforcement learning by first giving an overview in an intuitive way and then diving into proofs and solutions."
  />
</svelte:head>

<h1>Intuition</h1>
<Question>How can we start learning reinforcement learning?</Question>
<div class="separator" />

<svg version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter
      id="filter19205"
      x="-.05603"
      y="-.12606"
      width="1.1121"
      height="1.2526"
      color-interpolation-filters="sRGB"
    >
      <feGaussianBlur stdDeviation="6.5415986" />
    </filter>
  </defs>
  <g>
    <path
      d="m250 180-95-25"
      fill="none"
      stroke="var(--text-color)"
      stroke-linecap="round"
      stroke-width="1.5607"
    />
    <rect
      x="10"
      y="4"
      width="480"
      height="192"
      ry="14.4"
      fill="none"
      stroke="var(--text-color)"
      stroke-linecap="square"
      stroke-width="0.2px"
    />
    <text
      transform="rotate(-38.518)"
      x="-60.268604"
      y="125.71961"
      fill="var(--text-color)"
      font-family="sans-serif"
      font-size="40px"
      stroke="var(--text-color)"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="-60.268604" y="125.71961">Intuition</tspan></text
    >
    <text
      transform="rotate(37.953)"
      x="317.59506"
      y="-183.02377"
      fill="var(--text-color)"
      font-family="sans-serif"
      font-size="40px"
      stroke="var(--text-color)"
      style="line-height:1.25"
      xml:space="preserve"
      ><tspan x="317.59506" y="-183.02377">Math</tspan></text
    >
    <g fill="none" stroke="var(--text-color)">
      <path
        d="m395.17 183.51a146.3 146.3 0 0 0-145.17-128.51v0a146.3 146.3 0 0 0-144.98 128.5"
        stroke-linecap="square"
      />
      <g stroke-width="0.2px">
        <path d="m250 59v25" />
        <path d="m140 110 15 15" />
        <path d="m360 110-15 15" />
        <path d="m185 75 11.017 25" />
        <path d="m319.02 77-11.017 25" />
        <path d="m115 170h20" />
        <path d="m365 170h20" />
      </g>
    </g>
    <circle
      cx="250"
      cy="179.77"
      r="10"
      fill="var(--text-color)"
      stroke="var(--text-color)"
      stroke-linecap="square"
    />
    <path
      transform="matrix(.99096 0 0 1.017 2.2609 -2.0281)"
      d="m395.17 183.51a146.3 146.3 0 0 0-145.17-128.51v0a146.3 146.3 0 0 0-144.98 128.5"
      fill="none"
      filter="url(#filter19205)"
      stroke="var(--text-color)"
      stroke-linecap="square"
      style="mix-blend-mode:normal"
    />
  </g>
</svg>

<p>
  A big chunk of machine learning is mathematics and reinforcement learning is
  no exception. Often we can lose track of what we are actually trying to solve
  and lose ourselves in the definitions and mathematics. This chapter is inteded
  to give an intuitive introduction into the topic of reinforcement learning
  without any rigourous proofs. The intention of the chapter is to elicit
  excitement and passion and to show that under all those calculations lie
  simple but powerful ideas. Still all the knowledge learned in this chapter is
  going to be useful for when we start dealing with math and programming.
</p>
<div class="separator" />
