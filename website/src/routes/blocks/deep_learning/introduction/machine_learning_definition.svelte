<script>
  import { tweened } from "svelte/motion";
  import { draw } from "svelte/transition";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import Button from "$lib/Button.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Hightlight from "$lib/Highlight.svelte";

  import TraditionalParadigm from "./_ml_definition/TraditionalParadigm.svelte";
  import MlParadigm from "./_ml_definition/MlParadigm.svelte";

  let notes = [
    "This definition is supposedly based on Arthur Samuel (1959). The exact quote is not contained in any of his papers, only the general sentiment regarding that definition.",
  ];

  let references = [
    {
      title: "Some Studies in Machine Learning Using the Game of Checkers",
      author: "Samuel A.L",
      journal: "IBM Journal of Research and Development",
      volume: 44,
      pages: "206-226",
      year: 1959,
    },
  ];

  let disabledNormal = false;
  const xTranslateNormal = tweened(0, {
    duration: 400,
  });

  const x1Normal = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const x2Normal = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const inputOpacityNormal = tweened(1, {
    duration: 400,
    delay: 500,
  });

  const processOpacityNormal = tweened(1, {
    duration: 400,
  });

  async function handleNormalProgramming() {
    disabledNormal = true;
    await xTranslateNormal.set(175);
    await processOpacityNormal.set(0);
    await processOpacityNormal.set(1);
    await x1Normal.set(80);
    await x2Normal.set(40);
    await xTranslateNormal.set(340);
    await inputOpacityNormal.set(0);
    x1Normal.set(60);
    x2Normal.set(60);
    await xTranslateNormal.set(0);
    await inputOpacityNormal.set(1);
    disabledNormal = false;
  }

  let disabledML = false;
  let showImprovement = false;
  let step = 0;
  let maxSteps = 3;
  let learned1 = false;
  let learned2 = false;
  let learned3 = false;

  const xTranslateML = tweened(0, {
    duration: 400,
  });

  const x1ML = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const x2ML = tweened(60, {
    duration: 400,
    delay: 500,
  });

  const inputOpacityML = tweened(1, {
    duration: 400,
    delay: 500,
  });

  const processOpacityML = tweened(1, {
    duration: 400,
  });

  async function handleMLProgramming() {
    disabledML = true;
    step += 1;

    await xTranslateML.set(175);
    await processOpacityML.set(0);
    await processOpacityML.set(1);
    let y = ((80 - 60) * step) / maxSteps;
    await x1ML.set(60 + y);
    await x2ML.set(60 - y);
    await xTranslateML.set(340);
    if (step !== 3) {
      showImprovement = true;
    }
    if (step === 1) {
      learned1 = true;
    }
    if (step === 2) {
      learned2 = true;
      learned3 = true;
    }
    await inputOpacityML.set(0);
    showImprovement = false;
    x1ML.set(60);
    x2ML.set(60);
    await xTranslateML.set(0);
    await inputOpacityML.set(1);

    if (step === 3) {
      step = 0;
      learned1 = false;
      learned2 = false;
      learned3 = false;
    }
    disabledML = false;
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Machine Learning</title>
  <meta
    name="description"
    content="Machine learning is a programming paradigm, where the logic of a program is learned from data."
  />
</svelte:head>

<Container>
  <h1>Machine Learning</h1>
  <div class="separator" />
  <p>
    When books and lectures discuss the origin of machine learning, the name
    Arthur Lee Samuel is surely to come up. Arthur Samuel was one of the
    pioneers in the area of artificial intelligence, who is most known for his
    computer checkers program<InternalLink type={"reference"} id={1} />. He is
    also the person who coined the term
    <strong>Machine Learning</strong>
    and is responsible for its' most famous definition:
  </p>
  <p class="info">
    "Machine learning is the field of study that gives computers the ability to
    learn without being explicitly programmed
    <InternalLink type="note" id={1} />".
  </p>
  <p>
    While the above definition is commonly used, it is not the one that we find
    most useful. Throughout this block we will rely on a programming oriented
    definition of machine learning.
  </p>
  <p class="info">
    Machine learning is a programming paradigm where the logic of a program is
    automatically learned from data.
  </p>

  <h2>Programming Paradigms</h2>
  <p>
    While the term programming can be defined in a very broad way, for our
    purposes we will narrow it down to <span class="info"
      >writing functions in a programming language to solve a particular task.
    </span>
  </p>
  <p>
    When the programmer uses the traditional programming paradigm to solve a
    problem, he usually takes the following steps. He studies the problem,
    writes the first draft of the code and runs the function. If the output of
    the function corresponds to the expectations of the programmer for a
    particular set of inputs his job is done. If not, the programmer keeps
    improving the code of the function until the outputs of the function are
    satisfactory.
  </p>
  <TraditionalParadigm />

  <p>
    The machine learning approach is different in several aspects. While the
    programmer still needs to write some parts of the function explicitly, many
    parts of the logic of the function are adjusted in an automatic procedure.
    The function requires a set of inputs and the corresponding (correct)
    outputs for those inputs in order to be able to learn. The function takes
    those inputs in, uses so called
    <Hightlight>weights</Hightlight> to transform the inputs into the outputs and
    compares the expected outputs to those produced by the function. Using the differences
    between the actual and the produced outputs the weights are adjusted automatically
    to improve the outputs of the function.
  </p>
  <MlParadigm />

  <h2>Example</h2>
  <p>
    Let us use a stylized geometric example to demonstrate how classical
    programming paradigms differ from machine learning programming. Let us
    assume we are assigned a task to transform a triangle into a rectangle of
    same height.
  </p>

  <svg version="1.1" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
    <g fill="none" stroke="var(--text-color)">
      <path d="m30.784 21.587 24.216 48.404h-48.433z" />
      <rect x="345.98" y="20.98" width="49.02" height="49.02" />
      <path d="m60 45.49h275" stroke-dasharray="2, 2" />
    </g>
  </svg>

  <p>
    In classical programming the programmer could for example notice, that both
    shapes are polygons. The input (the triangle) has 3 connected points while
    the output (the rectangle) has 4 connected points. To complete the
    transformation from the triangle to the rectangle, we would need to add an
    additional point at the tip of the triangle and to pull the two points
    apart.
  </p>
  <svg version="1.1" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="marker2143" overflow="visible" orient="auto">
        <path
          transform="scale(.4) translate(7.4 1)"
          d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
          fill="var(--main-color-1)"
          fill-rule="evenodd"
          stroke="var(--text-color-1)"
          stroke-width="1pt"
        />
      </marker>
      <marker id="DotM" overflow="visible" orient="auto">
        <path
          transform="scale(.4) translate(7.4 1)"
          d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
          fill="var(--main-color-1)"
          fill-rule="evenodd"
          stroke="var(--main-color-1)"
          stroke-width="1pt"
        />
      </marker>
    </defs>
    <g fill="none" stroke="var(--text-color)">
      <path
        d="m30.784 21.587 24.216 48.404h-48.433z"
        marker-end="url(#marker2143)"
        marker-mid="url(#DotM)"
        marker-start="url(#marker2143)"
        stroke-width=".96837"
      />
      <path d="m60 45.49h275" stroke-dasharray="2, 2" />
      <path
        d="m345 20h50v49.991l-50 0.008767z"
        marker-end="url(#marker2143)"
        marker-mid="url(#DotM)"
        marker-start="url(#marker2143)"
        stroke-width=".96837"
      />
    </g>
  </svg>

  <p>
    After planning the logic of the program, he implements it in a programming
    language. In our example we represent the logic of the program using the
    following "programming board". The difference between different
    implementations lies in the activated (white) blocks of the board. Different
    activations would mean different logic. The below board exempliefies a logic
    that is suited to fully transform a triangle into a rectangle.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg
      version="1.1"
      viewBox="0 0 110 90"
      width="200px"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g transform="translate(-144.75,-10.25)" stroke="var(--text-color)">
        <rect x="149" y="12.5" width="102" height="85" fill="none" />
        <rect
          x="153.25"
          y="16.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect x="170.25" y="16.75" width="8.5" height="8.5" fill="none" />
        <rect x="187.25" y="16.75" width="8.5" height="8.5" fill="none" />
        <rect
          x="204.25"
          y="16.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect x="221.25" y="16.75" width="8.5" height="8.5" fill="none" />
        <rect
          x="238.25"
          y="16.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect x="153.25" y="33.75" width="8.5" height="8.5" fill="none" />
        <rect x="170.25" y="33.75" width="8.5" height="8.5" fill="none" />
        <rect
          x="187.25"
          y="33.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <g fill="none">
          <rect x="204.25" y="33.75" width="8.5" height="8.5" />
          <rect x="221.25" y="33.75" width="8.5" height="8.5" />
          <rect x="238.25" y="33.75" width="8.5" height="8.5" />
          <rect x="153.25" y="50.75" width="8.5" height="8.5" />
        </g>
        <rect
          x="170.25"
          y="50.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <g fill="none">
          <rect x="187.25" y="50.75" width="8.5" height="8.5" />
          <rect x="204.25" y="50.75" width="8.5" height="8.5" />
          <rect x="221.25" y="50.75" width="8.5" height="8.5" />
        </g>
        <rect
          x="238.25"
          y="50.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect
          x="153.25"
          y="67.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect x="170.25" y="67.75" width="8.5" height="8.5" fill="none" />
        <rect x="187.25" y="67.75" width="8.5" height="8.5" fill="none" />
        <rect
          x="204.25"
          y="67.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <g fill="none">
          <rect x="221.25" y="67.75" width="8.5" height="8.5" />
          <rect x="238.25" y="67.75" width="8.5" height="8.5" />
          <rect x="153.25" y="84.75" width="8.5" height="8.5" />
          <rect x="170.25" y="84.75" width="8.5" height="8.5" />
        </g>
        <rect
          x="187.25"
          y="84.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect
          x="204.25"
          y="84.75"
          width="8.5"
          height="8.5"
          fill="var(--main-color-3)"
        />
        <rect x="221.25" y="84.75" width="8.5" height="8.5" fill="none" />
        <rect x="238.25" y="84.75" width="8.5" height="8.5" fill="none" />
      </g>
    </svg>
  </SvgContainer>

  <p>
    The interactive example below shows the functionality of a program, where
    the logic is hardcoded by the developer. The "logic board" is fixed and is
    suited to transform a triangle into a rectangle.
  </p>
  <svg version="1.1" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
    <g stroke="var(--text-color)">
      <g id="logic">
        <g opacity={$processOpacityNormal} fill="#fff">
          <rect x="153.25" y="105" width="8.4993" height="8.5008" />
          <rect x="204.25" y="105" width="8.4993" height="8.5008" />
          <rect x="238.24" y="105" width="8.4993" height="8.5008" />
          <rect x="187.25" y="122" width="8.4993" height="8.5008" />
          <rect x="170.25" y="139" width="8.4993" height="8.5008" />
          <rect x="238.24" y="139" width="8.4993" height="8.5008" />
          <rect x="153.25" y="156" width="8.4993" height="8.5008" />
          <rect x="204.25" y="156" width="8.4993" height="8.5008" />
          <rect x="187.25" y="173.01" width="8.4993" height="8.5008" />
          <rect x="204.25" y="173.01" width="8.4993" height="8.5008" />
        </g>
        <g fill="none">
          <rect x="149" y="100.75" width="101.99" height="85.008" />
          <rect x="170.25" y="105" width="8.4993" height="8.5008" />
          <rect x="187.25" y="105" width="8.4993" height="8.5008" />
          <rect x="221.24" y="105" width="8.4993" height="8.5008" />
          <rect x="153.25" y="122" width="8.4993" height="8.5008" />
          <rect x="170.25" y="122" width="8.4993" height="8.5008" />
          <rect x="204.25" y="122" width="8.4993" height="8.5008" />
          <rect x="221.24" y="122" width="8.4993" height="8.5008" />
          <rect x="238.24" y="122" width="8.4993" height="8.5008" />
          <rect x="153.25" y="139" width="8.4993" height="8.5008" />
          <rect x="187.25" y="139" width="8.4993" height="8.5008" />
          <rect x="204.25" y="139" width="8.4993" height="8.5008" />
          <rect x="221.24" y="139" width="8.4993" height="8.5008" />
          <rect x="221.24" y="156" width="8.4993" height="8.5008" />
          <rect x="238.24" y="156" width="8.4993" height="8.5008" />
          <rect x="153.25" y="173.01" width="8.4993" height="8.5008" />
          <rect x="170.25" y="173.01" width="8.4993" height="8.5008" />
          <rect x="170.25" y="156" width="8.4993" height="8.5008" />
          <rect x="187.25" y="156" width="8.4993" height="8.5008" />
          <rect x="221.24" y="173.01" width="8.4993" height="8.5008" />
          <rect x="238.24" y="173.01" width="8.4993" height="8.5008" />
        </g>
      </g>
      <rect
        id="transformer"
        x="150"
        y="19.992"
        width="99.992"
        height="75.007"
        fill="none"
      />
      <g
        opacity={$inputOpacityNormal}
        transform="translate({$xTranslateNormal}, 0)"
      >
        <polygon fill="none" points="1,{$x1Normal} 1,{$x2Normal} 51,40 51,80" />
      </g>
    </g>
  </svg>

  <Button
    on:click={handleNormalProgramming}
    disabled={disabledNormal}
    value="RUN"
  />

  <p>
    In machine learning on the other hand the role of a developer is not to find
    the logic that can transform triangles into squares, but to design a program
    that can learn the logic which can transform triangles into squares.
  </p>

  <p>
    For that purpose the developer has access to a dataset that shows how for
    each of the inputs a corresponding output should look like.
  </p>

  <SvgContainer maxWidth="500px">
    <svg version="1.1" viewBox="0 0 315 150" xmlns="http://www.w3.org/2000/svg">
      <g fill="none" stroke="var(--text-color)">
        <path d="m30.784 5 24.216 48.404h-48.433z" stroke-width=".96837" />
        <rect
          x="5.9804"
          y="95"
          width="49.02"
          height="49.02"
          stroke-width=".98039"
        />
        <path
          d="m30 60v30"
          stroke-dasharray="2.19089, 4.38178"
          stroke-width="1.0954"
        />
        <g transform="translate(15.181)">
          <path d="m100 55-24.216-48.404h48.433z" stroke-width=".96837" />
          <rect
            x="75.98"
            y="95"
            width="49.02"
            height="49.02"
            stroke-width=".98039"
          />
          <path
            d="m100 60v30"
            stroke-dasharray="2.19089, 4.38178"
            stroke-width="1.0954"
          />
        </g>
        <g transform="translate(29.724)">
          <path d="m146.6 29.216 48.404-24.216v48.433z" stroke-width=".96837" />
          <rect
            x="145.98"
            y="95"
            width="49.02"
            height="49.02"
            stroke-width=".98039"
          />
          <path
            d="m170 60v30"
            stroke-dasharray="2.19089, 4.38178"
            stroke-width="1.0954"
          />
        </g>
        <g transform="translate(45)">
          <path
            d="m263.4 29.216-48.404 24.216v-48.433z"
            stroke-width=".96837"
          />
          <rect
            x="215.98"
            y="95"
            width="49.02"
            height="49.02"
            stroke-width=".98039"
          />
          <path
            d="m240 60v30"
            stroke-dasharray="2.19089, 4.38178"
            stroke-width="1.0954"
          />
        </g>
      </g>
    </svg>
  </SvgContainer>

  <p>
    The inintial logic of the program is going to be very far off from the
    correct logic. The programmer feeds the program with data and the algorithm
    measures the magnitude of the error. For example if the program produces a
    triangle (input = output) the logic is very far from the desired one. A
    trapezoid on the other hand is closer to a rectangle. The measure of the
    error is eventually used to automatically adjust the logic of the program to
    improve the performance. How the process exactly works depends on the
    algorithm, but the idea is that each iteration of data input, error
    measurement and logic adjustment leads to better and better performance
    until our goal is withing some boundary.
  </p>

  <SvgContainer maxWidth={"500px"}>
    <svg
      version="1.1"
      viewBox="0 0 120 60"
      width="200"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g fill="none" stroke="var(--text-color)" stroke-width=".96837">
        <path d="m27.96 6.0969 24.216 48.404h-48.433z" />
        <path d="m82.177 6.0969h20l14.216 48.404h-48.433z" />
      </g>
      <g
        fill="none"
        stroke="var(--main-color-1)"
        stroke-dasharray="1.96079, 3.92157"
        stroke-width=".98039"
      >
        <rect x="3.382" y="5.4902" width="49.02" height="49.02" />
        <rect x="67.598" y="5.4902" width="49.02" height="49.02" />
      </g>
    </svg>
  </SvgContainer>
  <p>
    Below is an interactive example where the machine learning process is shown.
    It takes the algorithm three iterations to learn the desired logic. At first
    the difference between the square and the produced output is relatively
    large, but the error is used to improve the logic of the program. In the
    third iteration the program produces the desired results (after that the
    example is reset).
  </p>
  <svg version="1.1" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
    <g stroke="var(--text-color)">
      <g id="logic">
        <g opacity={$processOpacityML} fill="#fff">
          <rect
            x="153.25"
            y="105"
            width="8.4993"
            height="8.5008"
            fill={learned1 === true ? "var(--main-color-2)" : "none"}
          />
          <rect
            x="204.25"
            y="105"
            width="8.4993"
            height="8.5008"
            fill={learned2 === true ? "var(--main-color-2)" : "none"}
          />
          <rect x="238.24" y="105" width="8.4993" height="8.5008" />
          <rect x="187.25" y="122" width="8.4993" height="8.5008" />
          <rect x="170.25" y="139" width="8.4993" height="8.5008" />
          <rect x="238.24" y="139" width="8.4993" height="8.5008" />
          <rect
            x="153.25"
            y="156"
            width="8.4993"
            height="8.5008"
            fill={learned3 === true ? "var(--main-color-2)" : "none"}
          />
          <rect x="204.25" y="156" width="8.4993" height="8.5008" />
          <rect x="187.25" y="173.01" width="8.4993" height="8.5008" />
          <rect x="204.25" y="173.01" width="8.4993" height="8.5008" />
        </g>
        <g fill="none">
          <rect x="149" y="100.75" width="101.99" height="85.008" />
          <rect x="170.25" y="105" width="8.4993" height="8.5008" />
          <rect x="187.25" y="105" width="8.4993" height="8.5008" />
          <rect x="221.24" y="105" width="8.4993" height="8.5008" />
          <rect x="153.25" y="122" width="8.4993" height="8.5008" />
          <rect x="170.25" y="122" width="8.4993" height="8.5008" />
          <rect x="204.25" y="122" width="8.4993" height="8.5008" />
          <rect x="221.24" y="122" width="8.4993" height="8.5008" />
          <rect x="238.24" y="122" width="8.4993" height="8.5008" />
          <rect x="153.25" y="139" width="8.4993" height="8.5008" />
          <rect x="187.25" y="139" width="8.4993" height="8.5008" />
          <rect x="204.25" y="139" width="8.4993" height="8.5008" />
          <rect x="221.24" y="139" width="8.4993" height="8.5008" />
          <rect x="221.24" y="156" width="8.4993" height="8.5008" />
          <rect x="238.24" y="156" width="8.4993" height="8.5008" />
          <rect x="153.25" y="173.01" width="8.4993" height="8.5008" />
          <rect x="170.25" y="173.01" width="8.4993" height="8.5008" />
          <rect x="170.25" y="156" width="8.4993" height="8.5008" />
          <rect x="187.25" y="156" width="8.4993" height="8.5008" />
          <rect x="221.24" y="173.01" width="8.4993" height="8.5008" />
          <rect x="238.24" y="173.01" width="8.4993" height="8.5008" />
        </g>
      </g>
      <rect
        id="transformer"
        x="150"
        y="19.992"
        width="99.992"
        height="75.007"
        fill="none"
      />

      <g
        opacity="1"
        stroke="var(--main-color-1)"
        stroke-dasharray="4 2"
        transform="translate(340, 0)"
      >
        <polygon fill="none" points="1,80 1,40 51,40 51,80" />
      </g>

      <g opacity={$inputOpacityML} transform="translate({$xTranslateML}, 0)">
        <polygon fill="none" points="1,{$x1ML} 1,{$x2ML} 51,40 51,80" />
      </g>

      {#if showImprovement}
        <path
          in:draw={{ duration: 1000 }}
          d="M 365 90 V 150 H 260"
          stroke="red"
          fill="none"
        />
      {/if}
    </g>
  </svg>
  <Button on:click={handleMLProgramming} disabled={disabledML} value="RUN" />

  <p>
    The general intuition about machine learning that you should keep in mind is
    as follows.
  </p>
  <p class="info">
    In classical programming and machine learning we try to solve a problem, by
    designing computer functions. In classical programming the programmer
    hardcodes the logic of that function. In machine learning the programmer
    chooses the algorithm and the parameters that are used to learn the
    function.
  </p>

  <p>
    One final question that we should ask ourselves before we move on to the
    next chapter is: when do we use machine learning and when do we use
    classical programming? Machine learning is usually used when the complexity
    of the program would get out of hand if we implemented the logic manually. A
    program that is able to recognize digits is almost impossible to implement
    by hand. How would you for example implement a program that is able to
    differentiate between an 8 and a 9? This is an especially hard problem when
    the location of the numbers is scattered and not centered in the middle of
    an image. The same problem can be solved relatively straightforward using
    neural networks, provided we have the necessary data.
  </p>
</Container>

<Container>
  <Footer {notes} {references} />
</Container>
