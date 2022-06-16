<script>
  import Highlight from "$lib/Highlight.svelte";
  import Sentence from "$lib/Sentence.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
</script>

<svelte:head>
  <title>World4AI | Books</title>
  <meta
    name="description"
    content="Jupyter notebooks contain implementations of the theory that is covered in the blocks section. The books contain topics such as Python, PyTorch, NumPy, Matplotlib, Pandas and so on."
  />
</svelte:head>

<h1>Jupyter Books</h1>
<div class="separator" />

<Container>
  <p>
    The online books that are presented below are designed to provide an
    opportunity to practice the theory that was developed in the <Highlight
      >blocks</Highlight
    > section or to develop basic skills that are required to study the practical
    books.
  </p>
  <SvgContainer maxWidth={"700px"}>
    <Sentence sentence={"jupyter"} />
  </SvgContainer>
  <SvgContainer maxWidth={"500px"}>
    <Sentence sentence={"books"} />
  </SvgContainer>

  <div class="separator" />
  <div class="book">
    <div class="logo">
      <!-- SOURCE: https://commons.wikimedia.org/wiki/File:Python-logo-notext.svg -->
      <svg viewBox="0 0 110 110">
        <defs id="defs2171">
          <linearGradient id="linearGradient11301" inkscape:collect="always">
            <stop
              id="stop11303"
              offset="0"
              style="stop-color:#ffe052;stop-opacity:1"
            />
            <stop
              id="stop11305"
              offset="1"
              style="stop-color:#ffc331;stop-opacity:1"
            />
          </linearGradient>
          <linearGradient
            gradientUnits="userSpaceOnUse"
            y2="168.1012"
            x2="147.77737"
            y1="111.92053"
            x1="89.136749"
            id="linearGradient11307"
            xlink:href="#linearGradient11301"
            inkscape:collect="always"
          />
          <linearGradient id="linearGradient9515" inkscape:collect="always">
            <stop
              id="stop9517"
              offset="0"
              style="stop-color:#387eb8;stop-opacity:1"
            />
            <stop
              id="stop9519"
              offset="1"
              style="stop-color:#366994;stop-opacity:1"
            />
          </linearGradient>
          <linearGradient
            gradientUnits="userSpaceOnUse"
            y2="131.85291"
            x2="110.14919"
            y1="77.070274"
            x1="55.549179"
            id="linearGradient9521"
            xlink:href="#linearGradient9515"
            inkscape:collect="always"
          />
        </defs>
        <g
          inkscape:label="Calque 1"
          inkscape:groupmode="layer"
          id="layer1"
          transform="translate(-473.36088,-251.72485)"
        >
          <g id="g1894" transform="translate(428.42338,184.2561)">
            <path
              style="opacity:1;color:#000000;fill:url(#linearGradient9521);fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter;marker:none;marker-start:none;marker-mid:none;marker-end:none;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;visibility:visible;display:inline;overflow:visible"
              d="M 99.75,67.46875 C 71.718268,67.468752 73.46875,79.625 73.46875,79.625 L 73.5,92.21875 L 100.25,92.21875 L 100.25,96 L 62.875,96 C 62.875,96 44.9375,93.965724 44.9375,122.25 C 44.937498,150.53427 60.59375,149.53125 60.59375,149.53125 L 69.9375,149.53125 L 69.9375,136.40625 C 69.9375,136.40625 69.433848,120.75 85.34375,120.75 C 101.25365,120.75 111.875,120.75 111.875,120.75 C 111.875,120.75 126.78125,120.99096 126.78125,106.34375 C 126.78125,91.696544 126.78125,82.125 126.78125,82.125 C 126.78125,82.124998 129.04443,67.46875 99.75,67.46875 z M 85,75.9375 C 87.661429,75.937498 89.8125,78.088571 89.8125,80.75 C 89.812502,83.411429 87.661429,85.5625 85,85.5625 C 82.338571,85.562502 80.1875,83.411429 80.1875,80.75 C 80.187498,78.088571 82.338571,75.9375 85,75.9375 z "
              id="path8615"
            />
            <path
              id="path8620"
              d="M 100.5461,177.31485 C 128.57784,177.31485 126.82735,165.1586 126.82735,165.1586 L 126.7961,152.56485 L 100.0461,152.56485 L 100.0461,148.7836 L 137.4211,148.7836 C 137.4211,148.7836 155.3586,150.81787 155.3586,122.53359 C 155.35861,94.249323 139.70235,95.252343 139.70235,95.252343 L 130.3586,95.252343 L 130.3586,108.37734 C 130.3586,108.37734 130.86226,124.03359 114.95235,124.03359 C 99.042448,124.03359 88.421098,124.03359 88.421098,124.03359 C 88.421098,124.03359 73.514848,123.79263 73.514848,138.43985 C 73.514848,153.08705 73.514848,162.6586 73.514848,162.6586 C 73.514848,162.6586 71.251668,177.31485 100.5461,177.31485 z M 115.2961,168.8461 C 112.63467,168.8461 110.4836,166.69503 110.4836,164.0336 C 110.4836,161.37217 112.63467,159.2211 115.2961,159.2211 C 117.95753,159.2211 120.1086,161.37217 120.1086,164.0336 C 120.10861,166.69503 117.95753,168.8461 115.2961,168.8461 z "
              style="opacity:1;color:#000000;fill:url(#linearGradient11307);fill-opacity:1;fill-rule:nonzero;stroke:none;stroke-width:1;stroke-linecap:butt;stroke-linejoin:miter;marker:none;marker-start:none;marker-mid:none;marker-end:none;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1;visibility:visible;display:inline;overflow:visible"
            />
          </g>
        </g>
      </svg>
    </div>
    <p class="text">
      Python has become the de facto only programming language of machine
      learning, especially when we consider deep learning. If you want to use
      machine learning in practice or to work on deep learning reseach, you will
      need become procicient in Python. The <a
        class="yellow"
        href="https://python.world4ai.org"
        target="_blank">Python book</a
      > by World4AI provides an overview of essential Python.
    </p>
  </div>

  <p class="warning">The NumPy book has not yet been started!</p>
  <!-- The logo is taken from https://numpy.org/-->
  <div class="book">
    <div class="logo">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500"
        ><defs
          ><style>
            .cls-1 {
              fill: rgb(77, 171, 207);
            }
            .cls-2 {
              fill: rgb(77, 119, 207);
            }
          </style></defs
        ><g id="Layer_1" data-name="Layer 1"
          ><polygon
            class="cls-1"
            points="220.93 127.14 151.77 92.23 75.87 130.11 146.9 165.78 220.93 127.14"
          /><polygon
            class="cls-1"
            points="252.63 143.14 325.14 179.74 249.91 217.52 178.77 181.79 252.63 143.14"
          /><polygon
            class="cls-1"
            points="349.47 92.76 423.96 130.11 357.34 163.57 284.68 126.92 349.47 92.76"
          /><polygon
            class="cls-1"
            points="317.41 76.67 250.35 43.05 184.01 76.15 253.11 111 317.41 76.67"
          /><polygon
            class="cls-1"
            points="264.98 365.44 264.98 456.95 346.22 416.41 346.13 324.86 264.98 365.44"
          /><polygon
            class="cls-1"
            points="346.1 292.91 346.01 202.32 264.98 242.6 264.98 333.22 346.1 292.91"
          /><polygon
            class="cls-1"
            points="443.63 275.93 443.63 367.8 374.34 402.38 374.29 310.93 443.63 275.93"
          /><polygon
            class="cls-1"
            points="443.63 243.81 443.63 153.79 374.21 188.3 374.27 279.07 443.63 243.81"
          /><path
            class="cls-2"
            d="M236.3,242.6l-54.72-27.51V334s-66.92-142.39-73.12-155.18c-.8-1.65-4.09-3.46-4.93-3.9-12-6.3-47.16-24.11-47.16-24.11V360.89l48.64,26V277.08s66.21,127.23,66.88,128.62,7.32,14.8,14.42,19.51c9.46,6.26,50,30.64,50,30.64Z"
          /></g
        ></svg
      >
    </div>
    <p class="text">
      NumPy is the fundamental package of scientific computing in Python. It is
      the glue that holds all of the machine learning stack together. Pandas,
      Matplotlib, SciPy, Sklearn and many more rely on NumPy.
    </p>
  </div>

  <div class="book">
    <!--The logo is originally taken from: https://commons.wikimedia.org/wiki/File:PyTorch_logo_black.svg-->
    <div class="logo">
      <svg
        viewBox="0 0 95 95"
        xmlns="http://www.w3.org/2000/svg"
        xmlns:xlink="http://www.w3.org/1999/xlink"
        version="1.1"
      >
        <g id="group">
          <path
            id="Path"
            d="M63.1 28.3 L56.5 34.9 C67.3 45.7 67.3 63.1 56.5 73.7 45.7 84.5 28.3 84.5 17.7 73.7 6.9 62.9 6.9 45.5 17.7 34.9 L34.8 17.8 37.2 15.4 37.2 2.5 11.4 28.3 C-3 42.7 -3 65.9 11.4 80.3 25.8 94.7 49 94.7 63.1 80.3 77.5 65.8 77.5 42.7 63.1 28.3 Z"
            fill="#ee4c2c"
            fill-opacity="1"
            stroke="none"
          />
          <path
            id="Path-1"
            d="M55 21.9 C55 24.551 52.851 26.7 50.2 26.7 47.549 26.7 45.4 24.551 45.4 21.9 45.4 19.249 47.549 17.1 50.2 17.1 52.851 17.1 55 19.249 55 21.9 Z"
            fill="#ee4c2c"
            fill-opacity="1"
            stroke="none"
          />
        </g>
      </svg>
    </div>
    <p class="text">
      Deep Learning is one of the major blocks covered on this website. The
      companion <a
        class="yellow"
        href="https://deeplearning.world4ai.org"
        target="_blank">deep learning book</a
      > enhances the theoretical concepts and intuitions from this website and provides
      a practical Python course. For the most part NumPy and PyTorch are the main
      libraries used in the book. While Keras (with TensorFlow) or JAX are perfectly
      fine choices to learn deep learning, we will focus on PyTorch for the time
      being.
    </p>
  </div>

  <div class="separator" />
</Container>

<style>
  .book {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .logo {
    flex-basis: 50px;
    flex-grow: 0;
    flex-shrink: 1;
    margin-right: 30px;
  }

  .text {
    flex: 3;
  }
</style>
