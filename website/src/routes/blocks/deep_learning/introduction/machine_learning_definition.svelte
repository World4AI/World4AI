<script>
  import { tweened } from "svelte/motion";
  import { draw } from "svelte/transition";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import PlayButton from "$lib/PlayButton.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Hightlight from "$lib/Highlight.svelte";

  import Function from "./_ml_definition/Function.svelte";
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

  async function handleNormalProgramming() {
    disabledNormal = true;
    await xTranslateNormal.set(175);
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

  let weight1Adjustment = tweened(0, {
    duration: 400,
  });
  let weight2Adjustment = tweened(0, {
    duration: 400,
  });
  let weight3Adjustment = tweened(0, {
    duration: 400,
  });
  let weight4Adjustment = tweened(0, {
    duration: 400,
  });

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
      await weight1Adjustment.set((step + 1) * -10);
      await weight2Adjustment.set((step + 1) * 7);
      await weight3Adjustment.set((step + 1) * -5);
      await weight4Adjustment.set((step + 1) * 10);
    }

    await inputOpacityML.set(0);
    x1ML.set(60);
    x2ML.set(60);
    await xTranslateML.set(0);
    await inputOpacityML.set(1);

    if (step === 3) {
      step = 0;
      await weight1Adjustment.set(0);
      await weight2Adjustment.set(0);
      await weight3Adjustment.set(0);
      await weight4Adjustment.set(0);
    }
    disabledML = false;
    showImprovement = false;
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Machine Learning</title>
  <meta
    name="description"
    content="Machine learning is a programming paradigm, where the logic of a program is learned from data."
  />
</svelte:head>

<h1>Machine Learning</h1>
<div class="separator" />
<Container>
  <p>
    When books and university lectures on the topic of artificial intelligence
    mention the origin of machine learning, the name Arthur Lee Samuel is surely
    to come up. Arthur Samuel was one of the pioneers in the area of artificial
    intelligence, who coined the term
    <strong>Machine Learning</strong>
    <InternalLink type={"reference"} id={1} />
    and is responsible for its' most famous definition:
  </p>
  <p class="info">
    "Machine learning is the field of study that gives computers the ability to
    learn without being explicitly programmed
    <InternalLink type="note" id={1} />".
  </p>
  <p>
    While the above definition is commonly used, it is not the one that we find
    most useful. Throughout the deep learning block we will rely on a more
    programming oriented definition of machine learning.
  </p>
  <p class="info">
    Machine learning is a programming paradigm where the solution to a problem
    is automatically learned from data.
  </p>

  <div class="separator" />
  <h2>Programming Paradigms</h2>
  <p>
    The task of the programmer is to find a function, that can generate desired
    outputs, based on the inputs of the function. For example a programmer might
    be assigned a task to write a spam filter, where the function would classify
    the email as spam or ham based on the contents of the email, the email
    address, the email subject and some additional metadata. It does not matter
    whether the programmer uses a traditional programming paradigm or machine
    learning, the result of the task is essentially the same: a function that
    takes those inputs and produces the classification as the output.
  </p>
  <Function />
  <p>
    The big difference between the classical and the machine learning progamming
    paradigm is the way that function is derived.
  </p>
  <p>
    When programmers apply a traditional programming paradigm to create a spam
    filter, they will study the problem at hand and look at the inputs of the
    function. They could for example recognize that the words <em
      >money, rich and quick</em
    > are common in spam emails and write the first draft of the the function. If
    the output of the function corresponds to the expectations of the programmers
    the job is done. If not, the programmers would keep improving the code of the
    function until the outputs of the function are satisfactory. For example the
    programmers might be satisfied, once they are able to classify spam emails with
    an accuracy of 95%.
  </p>
  <TraditionalParadigm />

  <p>
    The machine learning approach is different in several aspects. While the
    programmer still needs to write some parts of the function explicitly, many
    parts of the logic of the function are adjusted in an automatic procedure.
    The function requires a set of inputs and the corresponding (correct)
    outputs for those inputs in order to be able to learn. The programmer
    basically needs a dataset with the corresponding flags that show if the text
    (input) corresponds to a spam email (output). The function takes those
    inputs in, uses so called
    <Hightlight>weights</Hightlight> to transform the inputs into the outputs and
    compares the expected outputs to those produced by the function. Using the differences
    between the actual outputs (spam or not) and the produced outputs the weights
    are adjusted automatically to improve the outputs of the function.
  </p>
  <p class="info">
    If you have trouble understanding the weights of a function, just imagine
    that those weights somehow correspond to the logic of the function. In
    machine learning the weights can be changed automatically by measureing some
    error and trying to redice that error. The adjustment of those weights is
    what we refer to as learning.
  </p>
  <MlParadigm />

  <p>
    While both paradigms produce a function, in machine learing we commonly tend
    to use the word <Hightlight>model</Hightlight> instead of function.
  </p>
  <div class="separator" />
  <h2>Example</h2>
  <p>
    Let us use a stylized geometric example to further demonstrate how classical
    programming paradigms differ from machine learning programming. Let us
    assume we are assigned a task to transform a triangle into a rectangle of
    same height.
  </p>
</Container>

<div class="background-yellow">
  <SvgContainer maxWidth={"900px"}>
    <svg version="1.1" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
      <g fill="none" stroke="var(--text-color)">
        <path d="m30.784 21.587 24.216 48.404h-48.433z" />
        <rect x="345.98" y="20.98" width="49.02" height="49.02" />
        <path d="m60 45.49h275" stroke-dasharray="2, 2" />
      </g>
    </svg>
  </SvgContainer>
</div>
<Container>
  <p>
    In classical programming the programmer could for example notice, that both
    shapes are polygons. The input (the triangle) has 3 connected points while
    the output (the rectangle) has 4 connected points. To complete the
    transformation from the triangle to the rectangle, the programmer would need
    to add an additional point at the tip of the triangle and to pull the two
    points apart.
  </p>
</Container>

<div class="background-blue">
  <SvgContainer maxWidth={"900px"}>
    <svg version="1.1" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="marker2143" overflow="visible" orient="auto">
          <path
            transform="scale(.4) translate(7.4 1)"
            d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
            fill="var(--main-color-1)"
            fill-rule="evenodd"
            stroke="context-stroke"
            stroke-width="1pt"
          />
        </marker>
      </defs>
      <g fill="none" stroke="#000">
        <path
          d="m30.784 21.587 24.216 48.404h-48.433z"
          marker-end="url(#marker2143)"
          marker-mid="url(#marker2143)"
          marker-start="url(#marker2143)"
          stroke-width=".96837"
        />
        <path d="m60 45.49h275" stroke-dasharray="2, 2" />
        <path
          d="m345 20h50v49.991l-50 0.008767z"
          marker-end="url(#marker2143)"
          marker-mid="url(#marker2143)"
          marker-start="url(#marker2143)"
          stroke-width=".96837"
        />
      </g>
    </svg>
  </SvgContainer>
</div>

<Container>
  <p>
    The developer would then use a traditional programming language like
    JavaSript, C or Python and hardcode the logic that creates the additional
    polygon and pulls the polygons apart. The program could for example look as
    shown in the interactive example below.
  </p>

  <PlayButton on:click={handleNormalProgramming} disabled={disabledNormal} />
  <SvgContainer maxWidth={"800px"}>
    <svg version="1.1" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="marker2143" overflow="visible" orient="auto">
          <path
            transform="scale(.4) translate(7.4 1)"
            d="m-2.5-1c0 2.76-2.24 5-5 5s-5-2.24-5-5 2.24-5 5-5 5 2.24 5 5z"
            fill="var(--main-color-1)"
            fill-rule="evenodd"
            stroke="context-stroke"
            stroke-width="1pt"
          />
        </marker>
      </defs>
      <g stroke="var(--text-color)">
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
          <polygon
            transform="translate(3, 0)"
            fill="none"
            marker-end="url(#marker2143)"
            marker-mid="url(#marker2143)"
            marker-start="url(#marker2143)"
            points="0,{$x1Normal} 0,{$x2Normal} 51,40 51,80"
          />
        </g>
      </g>
    </svg>
  </SvgContainer>

  <p>
    In machine learning on the other hand the role of a developer is not to find
    the logic that can transform triangles into squares, but to design a program
    that can learn the logic which can transform triangles into squares. In
    machine learning this logic is represented by so called <Hightlight
      >weights</Hightlight
    >. Each of the weights is simply a number that is used in the function
    internally to process the inputs (the triangle in our case) in order to
    generate the output of the function. Learning then means in our case:
    "adjusting the weights to get better and better at transforming triangles
    into rectangles". How exactly the transforming with weights and learning
    looks like is going to be covered in the next lectures.
  </p>
  <p>
    Below for example we have four different weights that we will utilize in
    order to solve the task of turning triangles into rectangles.
  </p>
  <SvgContainer maxWidth={"250px"}>
    <svg version="1.1" viewBox="0 0 100 85" xmlns="http://www.w3.org/2000/svg">
      <g id="weights" fill="none" stroke="#000">
        <rect x="5" y="5" width="15" height="75" fill="var(--main-color-1)" />
        <rect x="30" y="40" width="15" height="40" fill="var(--main-color-2)" />
        <rect x="55" y="25" width="15" height="55" fill="var(--main-color-3)" />
        <rect x="80" y="55" width="15" height="25" fill="var(--main-color-4)" />
      </g>
    </svg>
  </SvgContainer>
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
    The inintial logic (first set of weights) of the program is going to be very
    far off from the correct logic. The programmer feeds the program with data
    and the algorithm measures the magnitude of the error. For example if the
    program produces a triangle (input equals output) the logic is very far from
    the desired one. A trapezoid on the other hand is closer to a rectangle. The
    measure of the error is eventually used to automatically adjust the weights
    of the program to improve the performance. How the process exactly works
    depends on the algorithm, but the idea is that each iteration of data input,
    error measurement and weight adjustment leads to better and better
    performance until the error is less than some threshold.
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
    large, but the error is used to adjust the weights of the function. In the
    third iteration the program produces the desired results (after that the
    example is reset).
  </p>
  <PlayButton on:click={handleMLProgramming} disabled={disabledML} />
  <svg version="1.1" viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
    <text x="10" y="20" style="line-height:1.25" xml:space="preserve"
      ><tspan x="0" y="160">Iteration Nr. {step}</tspan></text
    >
    <g stroke="var(--text-color)">
      <g id="weights" stroke="var(--text-color)">
        <rect
          x="155"
          y={115 - $weight1Adjustment}
          width="15"
          height={75 + $weight1Adjustment}
          fill="var(--main-color-1)"
        />
        <rect
          x="180"
          y={150 - $weight2Adjustment}
          width="15"
          height={40 + $weight2Adjustment}
          fill="var(--main-color-2)"
        />
        <rect
          x="205"
          y={135 - $weight3Adjustment}
          width="15"
          height={55 + $weight3Adjustment}
          fill="var(--main-color-3)"
        />
        <rect
          x="230"
          y={165 - $weight4Adjustment}
          width="15"
          height={25 + $weight4Adjustment}
          fill="var(--main-color-4)"
        />
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

  <div class="separator" />
  <p>
    The general intuition about machine learning that you should keep in mind is
    as follows.
  </p>
  <p class="info">
    In classical programming and machine learning we try to solve a problem by
    generating computer functions. In classical programming the programmer
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

<style>
  .background-yellow {
    background-color: var(--main-color-3);
    margin: 15px 0;
  }

  .background-blue {
    background-color: var(--main-color-4);
    margin: 15px 0;
  }
</style>
