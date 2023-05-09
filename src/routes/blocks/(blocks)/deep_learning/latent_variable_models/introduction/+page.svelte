<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Number from "./Number.svelte";
  import Alert from "$lib/Alert.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";

  const notes = [
    "The word latent comes from the latin word lateo, which actually means to be hidden.",
  ];

  const faceCharacteristics = [
    { title: "Eye", strength: 2 },
    { title: "Skin", strength: 1 },
    { title: "Haircut", strength: 4 },
    { title: "Ear", strength: 3 },
    { title: "Mouth", strength: 2 },
    { title: "Smile", strength: 3 },
    { title: "Glasses", strength: 1 },
  ];

  const numberOne = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
  ];

  const numberTwo = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
  ];

  const numberThree = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
  ];
</script>

<svelte:head>
  <title>Latent Variables Models - World4AI</title>
  <meta
    name="description"
    content="Latent variables are the unobservable and the non measurable variables, that determine the characteristics of the observable data. Oftentimes the distribution of the data is too complex to be learned directly so we represent latent variables using well understood distributions and learn a mapping from latent variables to the actual data."
  />
</svelte:head>

<Container>
  <Alert type="warning">This chapter is early work in progress</Alert>
  <h1>Latent Variables</h1>
  <div class="separator" />
  <p>
    This chapter is dedicated to generative models called <Highlight
      >latent variable models</Highlight
    >. When we deal with those types of models we differentiate between two
    types of variables: observable variables and latent variables. Observable
    variables is the data we have been dealing with so far and it is the part of
    the data that we can actually measure. When we are dealing with MNIST for
    example, the pixel values of the digits are the observable variables. A
    number 1 would for example have higher pixel values in the middle of the
    drawing and lower values in the surrounding pixels.
  </p>
  <Number number={numberOne} />
  <p>
    Latent variables<InternalLink type="note" id={1} /> on the other hand are not
    observable and there is no way for us to directly measure those.
  </p>
  <Alert type="info"
    >Latent variables are the unobservable variables, that determine the
    characteristics of the observable data.</Alert
  >
  <p>
    When you try to imagine latent variables, think about the hidden
    characteristics of the data. Let's look at the two images of digits below
    and try to get additional intuition what hidden variables might be contained
    in the images.
  </p>
  <div class="flex flex-col md:flex-row">
    <Number number={numberTwo} />
    <Number number={numberThree} />
  </div>
  <p>
    Depending on your country of origin there are different ways to write the
    number 1. In the United States it is common to draw the number 1 as a
    straight line, while in many countries in the European Union, there is an
    additional line attached to the top. While we do not observe directly the
    country of origin of the people who drew the digit, we can still make an
    educated guess that they are most likely from different regions of the
    world. But you can not measure the country of origin directly from the
    picture, as this is a latent variable. There are many more latent variables
    that could be encoded in a digits dataset: the level of curviness, the
    cleanliness of the drawing or the tilt of the digit. None of them are
    directly observable or measurable.
  </p>
  <p>
    If you are dealing with human faces on the other hand the latent variables
    might be the color of the skin, the shape of the mouth, the gender, the
    haircut, glasses and so on.
  </p>
  <SvgContainer maxWidth="200px">
    <svg viewBox="0 0 300 300">
      {#each faceCharacteristics as characteristic, idx}
        <Block
          x="65"
          y={20 + idx * 43}
          width="120"
          height="35"
          text={characteristic.title}
          fontSize="25"
          class="fill-slate-200"
        />
        {#each Array(5) as _, colIdx}
          <Block
            x={150 + colIdx * 33}
            y={20 + idx * 43}
            width="20"
            height="20"
            class={colIdx <= characteristic.strength ? "fill-red-400" : "none"}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    So instead of generating the face directly, we could first sample the
    characteristics of the face from the distribution of latent variables and
    then generate the image based on those latent characteristics.
  </p>
  <p>
    While the above examples make it look like latent variables could be easily
    translated into human language, often latent variables are obscure and not
    easily interpretable. Still the above description should provide you with
    the necessary intuition that you will require during the following sections.
  </p>
</Container>
<Footer {notes} />
