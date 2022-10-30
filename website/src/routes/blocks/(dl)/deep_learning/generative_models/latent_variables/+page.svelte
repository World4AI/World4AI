<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Number from "./Number.svelte";
  import Alert from "$lib/Alert.svelte";
  import Latex from "$lib/Latex.svelte";

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
  ]


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
  ]

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
  ]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Latent Variables </title>
  <meta
    name="description"
    content="Latent variables are the unobservable and the non measurable variables, that determine the characteristics of the observable data."
  />
</svelte:head>

<h1>Latent Variables</h1>
<div class="separator"></div>

<Container>
  <p>In the next sections we will start working with so called <Highlight>latent variable models</Highlight>, so let's introduce latent variables, before we take a deep dive into those models.</p>
  <p>We need to differentiate between two types of variables: observable variables and latent variables. Observable variables is the data we have been dealing with so far and it is the part of the data that we can actually measure. When we are dealing with MNIST for example, the pixel values of the digits are the observable variables.</p>
  <Number number={numberOne} />
  <Alert type="info">Latent variables are the unobservable variables, that determine the characteristics of the observable data.</Alert>
  <p>Latent variables on the other hand are not observable. The word latent comes from the latin word 'lateo', which actually means 'to be hidden'. </p>
  <p>Let's look at the two images of digits below and try to get an intuition what hidden variables might be contained in the images.</p>
  <Number number={numberTwo} />
  <Number number={numberThree} />
  <p>Depending on your country of origin there are different ways to write the number 1. In the United States it is common to draw the number 1 as a straight line, while in many countries in the European Union, there is an additional line attached to the top. While we do not observe directly the country of origin of the people who made the image, we still can make an educated guess that they are most likely from different regions of the world. But you can not measure the country of origin directly from the picture, as this is a latent variable.</p>
  <p>There are many more latent variables that could be encoded in a digits dataset: the level of curviness, the cleanliness of the drawing or the tilt of the digit. None of them are directly observable or measurable.</p>

  <p>We usually use the letter <Latex>{String.raw`\mathbf{z}`}</Latex> to represent latent variables (actually vectors) and as usual we use the letter <Latex>{String.raw`\mathbf{X}`}</Latex> to represent the observable variables. When we use generative models we try to sample new data using the underlying probability distribution of the data <Latex>{String.raw`\mathbf{x} \sim p(\mathbf{X})`}</Latex>. But in a latent variable model we take a two step process.</p>
  <Latex>{String.raw`\mathbf{z} \sim p(\mathbf{z}) \\
                    \mathbf{X} \sim p(\mathbf{X} | \mathbf{z})
  `}</Latex> 

  <p>We first sample a latent variable, which determines the characteristics of the data to be drawn and in the second step we generate the data we are actually interested in, based on the latent variable.</p>
  <div class="separator"></div>

</Container>
