<svelte:head>
  <title>World4AI | Deep Learning | Introduction</title>
  <meta
    name="description"
    content="Deep learning is the main contributer of the recent success of machine learning."
  />
</svelte:head>

<h1>Deep Learning</h1>
<div class="separator" />

<svg
  version="1.1"
  width="500px"
  viewBox="0 0 500 500"
  xmlns="http://www.w3.org/2000/svg"
>
  <g stroke="var(--text-color)" stroke-width="0.2px">
    <g fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25 375 150-300" />
      <path d="m25 425 150-350" />
      <path d="m25 475 150-400" />
    </g>
    <g transform="translate(0,50)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25 375 150-300" />
      <path d="m25 425 150-350" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(0,100)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25 375 150-300" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(0,150)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(0,200)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(0,250)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25-225 150 300" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(0,300)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25-275 150 350" />
      <path d="m25-225 150 300" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(0,350)" fill="none">
      <path d="m25 25 150 50" />
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25-325 150 400" />
      <path d="m25-275 150 350" />
      <path d="m25-225 150 300" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
    </g>
    <g transform="translate(150)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25 375 150-300" />
      <path d="m25 425 150-350" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,50)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25 375 150-300" />
      <path d="m25 25 150 50" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,100)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25 325 150-250" />
      <path d="m25-25 150 100" />
      <path d="m25 25 150 50" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,150)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25 275 150-200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
      <path d="m25 25 150 50" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,200)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25 225 150-150" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
      <path d="m25 25 150 50" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,250)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25 175 150-100" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
      <path d="m25 25 150 50" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,300)" fill="none">
      <path d="m25 75h150" />
      <path d="m25 125 150-50" />
      <path d="m25-225 150 300" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
      <path d="m25 25 150 50" />
      <path d="m25 125 150-50" />
    </g>
    <g transform="translate(150,350)" fill="none">
      <path d="m25 75h150" />
      <path d="m25-225 150 300" />
      <path d="m25-175 150 250" />
      <path d="m25-125 150 200" />
      <path d="m25-75 150 150" />
      <path d="m25-25 150 100" />
      <path d="m25 25 150 50" />
      <path d="m25-275 150 350" />
    </g>
    <g fill="none">
      <path d="m325 75 150 160" />
      <path d="m325 125 150 110" />
      <path d="m325 175 150 60" />
      <path d="m325 225 150 10" />
      <path d="m325 275 150-40" />
      <path d="m325 325 150-90" />
      <path d="m325 375 150-140" />
      <path d="m325 425 150-190" />
    </g>
    <g transform="translate(0,50)" fill="none">
      <path d="m325 75 150 160" />
      <path d="m325 125 150 110" />
      <path d="m325 175 150 60" />
      <path d="m325 225 150 10" />
      <path d="m325 275 150-40" />
      <path d="m325 325 150-90" />
      <path d="m325 375 150-140" />
      <path d="m325 25 150 210" />
    </g>
    <g
      stroke="black"
      transform="translate(-40)"
      fill="var(--main-color-1)"
      stroke-width="1"
    >
      <rect x="350" y="60" width="30" height="30" />
      <rect x="350" y="110" width="30" height="30" />
      <rect x="350" y="160" width="30" height="30" />
      <rect x="350" y="210" width="30" height="30" />
      <rect x="350" y="260" width="30" height="30" />
      <rect x="350" y="310" width="30" height="30" />
      <rect x="350" y="360" width="30" height="30" />
      <rect x="350" y="410" width="30" height="30" />
    </g>
    <g stroke="black" fill="var(--main-color-1)" stroke-width="1">
      <rect x="160" y="60" width="30" height="30" />
      <rect x="160" y="110" width="30" height="30" />
      <rect x="160" y="160" width="30" height="30" />
      <rect x="160" y="210" width="30" height="30" />
      <rect x="160" y="260" width="30" height="30" />
      <rect x="160" y="310" width="30" height="30" />
      <rect x="160" y="360" width="30" height="30" />
      <rect x="160" y="410" width="30" height="30" />
    </g>
    <g stroke="black" fill="var(--main-color-1)" stroke-width="1">
      <rect x="10" y="10" width="30" height="30" />
      <rect x="10" y="60" width="30" height="30" />
      <rect x="10" y="110" width="30" height="30" />
      <rect x="10" y="160" width="30" height="30" />
      <rect x="10" y="210" width="30" height="30" />
      <rect x="10" y="260" width="30" height="30" />
      <rect x="10" y="310" width="30" height="30" />
      <rect x="10" y="360" width="30" height="30" />
      <rect x="10" y="410" width="30" height="30" />
      <rect x="10" y="460" width="30" height="30" />
    </g>
    <g stroke="black" fill="var(--main-color-1)" stroke-width="1">
      <rect x="460" y="220" width="30" height="30" />
      <rect x="460" y="270" width="30" height="30" />
    </g>
  </g>
</svg>
<p>
  In the last decade we drastically improved our ability to classify images, to
  generate text and to play video games with the help of machine learning. Deep
  learning was the main factor that drove those developments. In this block we
  are going to take a deep dive into artificial neural networks. We will study
  the very first neurons that were developed in the 50's, the progress towards
  neural networks and the backpropagation algorithm and the modern deep
  learning, that has started in 2010's and is still going strong.
</p>
<div class="separator" />
