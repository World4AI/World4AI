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
<div class="flex-center">
  <div class="container">
    <svg version="1.1" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter
          id="filter7566"
          x="-.15963"
          y="-.084419"
          width="1.3246"
          height="1.1688"
          color-interpolation-filters="sRGB"
        >
          <feGaussianBlur stdDeviation="6" />
        </filter>
      </defs>
      <g fill="none" stroke="#fff">
        <path
          d="m240 140c20.572-0.50795 41.25 6.3029 57.494 18.937 16.243 12.634 27.935 30.999 32.506 51.063 6.065 26.619-0.81242 55.879-18.1 77.009s-44.607 33.665-71.9 32.991"
          filter="url(#filter7566)"
          stroke-width="1px"
          style="mix-blend-mode:normal"
        />
        <g stroke-linecap="round" stroke-width="3">
          <path
            d="m250 92.595c-72.306 2.87e-4 -130.92 58.616-130.92 130.92 0.076 23.866 6.6571 46.905 18.603 66.853 14.404 24.053 34.032 59.659 34.032 86.788h147.9a8.6733 8.6733 135 0 0 8.6734-8.6734c0-22.339 19.628-54.062 34.032-78.115 11.946-19.948 18.527-42.987 18.603-66.853 3.7e-4 -72.306-58.615-130.92-130.92-130.92z"
          />
          <rect
            x="185.36"
            y="389.88"
            width="129.29"
            height="43.163"
            ry="10.792"
          />
          <rect
            x="203.57"
            y="445.77"
            width="92.855"
            height="37.302"
            ry="9.9911"
          />
          <rect
            transform="rotate(-90)"
            x="-239.06"
            y="27.001"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <rect
            transform="rotate(-60)"
            x="-82.589"
            y="110.95"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <rect
            transform="rotate(-30)"
            x="94.889"
            y="105.41"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <rect
            x="245.82"
            y="11.878"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <rect
            transform="rotate(30)"
            x="329.77"
            y="-144.59"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <rect
            transform="rotate(60)"
            x="324.23"
            y="-322.07"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <rect
            transform="rotate(90)"
            x="230.7"
            y="-473"
            width="8.3582"
            height="66.866"
            ry="4.1791"
          />
          <path
            d="m250 114.13c-61.355 2.4e-4 -111.09 49.746-111.09 111.11 0.056 17.607 4.2833 34.684 12.046 49.968 12.665 24.935 32.617 57.4 32.617 80.424h124.19a8.6733 8.6733 135 0 0 8.6734-8.6734c0-18.234 19.952-46.815 32.617-71.75 7.7628-15.284 11.99-32.361 12.046-49.968 3.1e-4 -61.365-49.737-111.11-111.09-111.11z"
          />
        </g>
        <path
          d="m240 140c20.572-0.50795 41.25 6.3029 57.494 18.937 16.243 12.634 27.935 30.999 32.506 51.063 6.065 26.619-0.81242 55.879-18.1 77.009s-44.607 33.665-71.9 32.991"
          stroke-width="1px"
        />
      </g>
    </svg>
  </div>
</div>
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

<style>
  .container {
    max-width: 300px;
    min-width: 200px;
  }
</style>
