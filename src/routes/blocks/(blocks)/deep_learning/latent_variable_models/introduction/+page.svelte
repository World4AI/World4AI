<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Number from "./Number.svelte";
  import Alert from "$lib/Alert.svelte";
  import Latex from "$lib/Latex.svelte";
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
  <title>World4AI | Deep Learning | Latent Variables</title>
  <meta
    name="description"
    content="Latent variables are the unobservable and the non measurable variables, that determine the characteristics of the observable data."
  />
</svelte:head>

<Container>
  <Alert type="warning">This chapter is early work in progress</Alert>
  <h1>Latent Variables</h1>
  <div class="separator" />
  <h2>Intuition</h2>
  <p />
  <p>
    In this section we will use so called <Highlight
      >latent variable models</Highlight
    > to generate images, so let's introduce latent variables, before we take a deep
    dive into those models.
  </p>
  <p>
    In latent variable models we differentiate between two types of variables:
    observable variables and latent variables. Observable variables is the data
    we have been dealing with so far and it is the part of the data that we can
    actually measure. When we are dealing with MNIST for example, the pixel
    values of the digits are the observable variables.
  </p>
  <Number number={numberOne} />
  <Alert type="info"
    >Latent variables are the unobservable variables, that determine the
    characteristics of the observable data.</Alert
  >
  <p>
    Latent variables on the other hand are not observable<InternalLink
      type="note"
      id={1}
    /> and there is no way for us to directly measure those. When you try to imagine
    latent variables, think about the characteristics of the data. If you are dealing
    with human faces, you could think about the color of the skin, the shape of the
    mouth, the gender, the haircut, glasses and so on.
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
        />
        {#each Array(5) as _, colIdx}
          <Block
            x={150 + colIdx * 33}
            y={20 + idx * 43}
            width="20"
            height="20"
            color={colIdx <= characteristic.strength
              ? "var(--main-color-1)"
              : "none"}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    So instead of generating the face directly, we could first sample the
    characteristics of the face and then sample the image based on those
    characteristics. It should be obvious that there is no direct mapping from
    latent variables to human understandable characteristics, but the intuition
    that the latent variables describe the data somehow should still make some
    sense.
  </p>
  <p>
    Let's look at the two images of digits below and try to get additional
    intuition what hidden variables might be contained in the images.
  </p>
  <Number number={numberTwo} />
  <Number number={numberThree} />
  <p>
    Depending on your country of origin there are different ways to write the
    number 1. In the United States it is common to draw the number 1 as a
    straight line, while in many countries in the European Union, there is an
    additional line attached to the top. While we do not observe directly the
    country of origin of the people who made the image, we still can make an
    educated guess that they are most likely from different regions of the
    world. But you can not measure the country of origin directly from the
    picture, as this is a latent variable. There are many more latent variables
    that could be encoded in a digits dataset: the level of curviness, the
    cleanliness of the drawing or the tilt of the digit. None of them are
    directly observable or measurable.
  </p>
  <div class="separator" />

  <h2>Mathematics</h2>
  <p>
    While latent variables make a lot of intuitive sense, the full understanding
    of latent variable models requires quite a lot of math. We will try to keep
    the intuition and the mathematical derivations separate, so that you could
    implement the models without the full knowledge of the background
    mathematics. If you struggle with the derivation, you could skip the
    mathematical sections, brush up on you probability skills (espesially
    Bayesian inference) and return at a later point, when you feel ready for the
    math.
  </p>
  <p>
    In this section we will introduce the basic notation and the general
    setting.
  </p>
  <p>
    We usually use the letter <Latex>{String.raw`\mathbf{z}`}</Latex> to represent
    latent variables (actually vectors) and as usual we use the letter <Latex
      >{String.raw`\mathbf{x}`}</Latex
    > to represent the observable variables.
  </p>
  <p>
    <Latex
      >{String.raw`
      \mathbf{z}: \text{Latent Variable} \\ 
      \mathbf{x}: \text{Observable Variable} \\ 
  `}</Latex
    >
  </p>
  <p>
    In latent variable models data generation is a two step process. We first
    sample a latent variable, which determines the characteristics of the data
    to be drawn and in the second step we generate the data we are actually
    interested in, based on the latent variable.
  </p>
  <Latex
    >{String.raw`1: \mathbf{z} \sim p_{\theta}(\mathbf{z}) \\
                    2: \mathbf{x} \sim p_{\theta}(\mathbf{x} | \mathbf{z})
  `}</Latex
  >

  <p>
    In order to train our generative model, we would like to maximize the
    log-likelihood.
  </p>
  <Latex
    >{String.raw`
      \log p_{\theta}(\mathbf{x}) = \log {\displaystyle\int} p_{\theta}(\mathbf{x} | \mathbf{z}) p_{\theta}(\mathbf{z})d\mathbf{z}
  `}</Latex
  >
  <p>
    Unfortunately this integral is usually intractable, given that we do not
    have a closed form solution and the dimensionality of the latent variable is
    usually quite high.
  </p>

  <!--
 <Latex>{String.raw`
  p(\mathbf{z} | \mathbf{x}) = \dfrac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{x})} = \dfrac{p(\mathbf{x}, \mathbf{z})}{\int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}}
 `}</Latex> 
 -->
</Container>
<Footer {notes} />
