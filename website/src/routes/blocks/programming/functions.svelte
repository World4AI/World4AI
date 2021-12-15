<script>
  import Question from "$lib/Question.svelte";
  import { onMount } from "svelte";

  onMount(() => {
    let interval = setInterval(() => {
      offset += 40;
    }, 300);

    return () => clearInterval(interval);
  });
  let offset = 0;
</script>

<svelte:head>
  <title>World4AI | Programming | Functions</title>
  <meta
    name="description"
    content="Python functions are used to avoid repeatable code and to divide large chunks of code into small pieces. Overall functions avoid repeatability and make the code more managable and maintainable."
  />
</svelte:head>

<h1>Functions</h1>
<Question>What are functions and why are they useful?</Question>
<div class="separator" />

<svg version="1.1" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="var(--text-color)">
    <path d="m440 105h60" stroke-width="6.1237" />
    <path d="m0 150h40" stroke-width="5" />
    <path
      d="m395 130c5.001 0 10.001 0 12.5-2.4996s2.4995-7.4994 6.6668-10 12.5-2.5005 20.833-2.5005"
      stroke-width="2"
    />
    <path
      d="m140 45c7.4085 0 14.816 0 18.519 7.5006s3.7032 22.5 8.3334 30 13.889 7.4995 23.148 7.4995"
      stroke-width="2"
    />
    <path
      d="m75 105c7.6798 3.9302 13.377 11.527 15 20 0.90395 4.7202 0.59974 9.5937-0.09297 14.349-0.69272 4.7557-1.7689 9.4507-2.4106 14.214-0.64177 4.7629-0.84104 9.6506 0.21401 14.339 1.055 4.6887 3.4626 9.1907 7.2896 12.098 2.8507 2.1654 6.4201 3.3674 10 3.3674s7.1493-1.202 10-3.3674"
      stroke-width="5"
    />
  </g>
  <g fill="var(--text-color)" stroke="#000">
    <rect x="125" y="35" width="15" height="160" ry="5" />
    <rect x="39" y="90" width="36" height="105" ry="7" stroke-width=".7746" />
    <rect x="436" width="4" height="200" ry="2" stroke-width=".57735" />
  </g>
  <g transform="matrix(.70588 0 0 .70588 152.65 48.529)" stroke="#000">
    <g fill="var(--text-color)">
      <rect x="205" y="30" width="25" height="170" ry="5" />
      <rect
        x="195"
        y="50"
        width="45"
        height="5"
        ry="2.5"
        stroke-width="1.1339"
      />
      <rect x="195" y="140" width="45" height="5" ry="2.5" />
    </g>
    <g fill="none" stroke-width="1px">
      <path d="m205 80h25" />
      <path d="m205 90h25" />
      <path d="m205 100h25" />
      <path d="m205 110h25" />
      <path d="m205 120h25" />
      <path d="m205 130h25" />
      <path d="m205 155h25" />
      <path d="m205 165h25" />
      <path d="m205 175h25" />
    </g>
    <rect
      x="190"
      y="190"
      width="55"
      height="10"
      ry="2.5"
      fill="var(--text-color)"
    />
    <path d="m205 35h25" fill="none" stroke-width="1px" />
    <path d="m205 40h25" fill="none" stroke-width="1px" />
  </g>
  <rect
    x="-.14086"
    y="189.83"
    width="500.28"
    height="10.313"
    ry="0"
    fill="var(--text-color)"
    stroke="#000"
    stroke-width=".71828"
  />
  <path
    d="m185 190 5-170h15l5 170z"
    fill="var(--text-color)"
    stroke="#000"
    stroke-width="1px"
  />
  <path
    d="m207 75h35v40h55"
    fill="none"
    stroke="var(--text-color)"
    stroke-width="5"
  />
  <g stroke="#000" stroke-width="1px">
    <path d="m39 99h36v3h-36z" />
    <path d="m39 112h36v3h-36z" />
    <path d="m39 127h36v3h-36z" />
  </g>
  <path
    d="m55 85h5v5h-5z"
    fill="var(--text-color)"
    stroke="#000"
    stroke-width="1px"
  />
  <rect
    x="115"
    y="115"
    width="55"
    height="75"
    ry="7"
    fill="var(--text-color)"
    stroke="#000"
  />
  <g fill="none">
    <g stroke="#000">
      <g stroke-width="1px">
        <path d="m125 55h15" />
        <path d="m125 65h15" />
        <path d="m125 75h15" />
        <path d="m125 85h15" />
        <path d="m125 95h15" />
        <path d="m125 105h15" />
      </g>
      <rect
        x="120"
        y="120"
        width="45"
        height="65"
        ry="5.4167"
        stroke-width="1.104"
      />
    </g>
    <path
      d="m305 70c0-13.899 0-27.799 5.0006-34.749 5.0006-6.9505 15-6.9505 20 17.517 4.9995 24.468 4.9995 73.4 11.667 97.866 6.6673 24.466 20 24.466 26.667 20.098 6.6662-4.3686 6.6662-13.107 6.6662-21.846"
      stroke="var(--text-color)"
      stroke-width="5"
    />
    <path d="m376 0v110" stroke="var(--text-color)" stroke-width="1px" />
  </g>
  <rect
    x="355"
    y="108"
    width="40"
    height="40"
    ry="5"
    fill="var(--text-color)"
    stroke="#000"
  />
  <rect
    x="358"
    y="111"
    width="34"
    height="34"
    ry="4"
    fill="var(--text-color)"
    stroke="#000"
    stroke-width="1.36"
  />
  <g fill="none" stroke="#000" stroke-width="1px">
    <path d="m190 170h15" />
    <path d="m190 165h15" />
    <path d="m190 160h15" />
    <path d="m190 155h15" />
  </g>
  <rect
    x="433.5"
    y="50"
    width="9"
    height="100"
    ry="5"
    fill="var(--text-color)"
    stroke="#000"
    stroke-width=".61237"
  />
  <path d="m155 150h5v5h-5z" fill="none" stroke="#000" stroke-width="1px" />
  <g fill="none" stroke="#000">
    <g stroke-dasharray="4,2,1,2" stroke-dashoffset={offset}>
      <path d="m440 105h60" />
      <path
        d="m75 105c7.6798 3.9302 13.377 11.527 15 20 0.90395 4.7202 0.59974 9.5937-0.09297 14.349-0.69272 4.7557-1.7689 9.4507-2.4106 14.214-0.64177 4.7629-0.84104 9.6506 0.21401 14.339 1.055 4.6887 3.4626 9.1907 7.2896 12.098 2.8507 2.1654 6.4201 3.3674 10 3.3674s7.1493-1.202 10-3.3674"
      />
      <path d="m0 150h40" />
      <path
        d="m140 45c7.4085 0 14.816 0 18.519 7.5006s3.7032 22.5 8.3334 30 13.889 7.4995 23.148 7.4995"
      />
      <path d="m207 75h35v40h55" />
      <path
        d="m305 70c0-13.899 0-27.799 5.0006-34.749 5.0006-6.9505 15-6.9505 20 17.517 4.9995 24.468 4.9995 73.4 11.667 97.866 6.6673 24.466 20 24.466 26.667 20.098 6.6662-4.3686 6.6662-13.107 6.6662-21.846"
      />
    </g>
    <path
      stroke-dashoffset={offset}
      d="m395 130c5.001 0 10.001 0 12.5-2.4996s2.4995-7.4994 6.6668-10 12.5-2.5005 20.833-2.5005"
      stroke-dasharray="4,2,1,2"
    />
  </g>
  <path d="m380 130h10" fill="none" stroke="#000" stroke-width="1px" />
  <path d="m380 125h10" fill="none" stroke="#000" stroke-width="1px" />
  <rect x="374" width="4" height="3" ry="0" fill="var(--text-color)" />
</svg>

<p>
  In programming we often face tasks that we have to repeat over and over again.
  Functions allow us to write the logic for these tasks only once and and to
  reuse them.
</p>
<p>
  The name comes from the idea that a single function has a role (a function) to
  fullfill. Large programs usually stack these functions on top of each other to
  solve a larger task. The smaller peaces are usually easier to maintain and to
  debug.
</p>
<p>
  Furthermore a Python function is often comparable to a mathematical function.
  The function (usually) takes an input and (usually) generates an output.
</p>
<div class="separator" />
