<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";

  import ButtonContainer from "$lib/button/ButtonContainer.svelte"; 
  import PlayButton from "$lib/button/PlayButton.svelte"; 

  import Latex from "$lib/Latex.svelte";

  const imageLength = 10;
  const pixelSize = 25;

  let activeRow = 0;
  let activeCol = 0;

  function f() {
    if (activeCol < imageLength-1) {
      activeCol += 1;
    } else if(activeRow < imageLength - 1) {
      activeCol = 0;
      activeRow += 1;
    } else {
      activeRow = 0;
      activeCol = 0;
    }
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Autoregressive Generative Models </title>
  <meta
    name="description"
    content="Autoregressive generative models, like GPT, generate one element at a time, where the next element depends on all previously generated elements and the model is not allowed to look at the future tokens."
  />
</svelte:head>

<h1>Autoregressive Generative Models</h1>
<div class="separator"></div>

<Container>
  <p>We have already encountered autoregressive models, when we were dealing with language models, like GPT. Simply put an autoregressive model relies on its previous values to generate the next value. So if you want to produce the fifth word in a sentence, you provide the model with the previous four words. In this section we will generate images one pixel at a time using autoregressive generative models, but you can apply the same ideas to text, audio and much more.</p>
  <ButtonContainer>
    <PlayButton {f} delta={70} />
  </ButtonContainer>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 300">
      {#each Array(imageLength) as _, colIdx}
        {#each Array(imageLength) as _, rowIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={2 + rowIdx * (pixelSize + 5)} 
                width={pixelSize} 
                height={pixelSize} 
                fill={colIdx === activeCol && rowIdx === activeRow ? "var(--main-color-1)" : 
                rowIdx < activeRow || (rowIdx===activeRow && colIdx < activeCol ) 
                ? "var(--main-color-3)" : "none"} 
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>The interactive example above shows how pixel generation looks like. To generate the red pixel, the model is allowed to look at all previous (yellow) pixels. Looking ahead is obviously not allowed, as this would allow the model to look at future pixels that were not produced yet, which would prevent the model from learning to generate pixels based on the past.</p>

  <p>In mathematical terms an autoregressive model calculates the probability  of some image vector <Latex>{String.raw`\mathbf{x}`}</Latex> (like an image), by using the chain rule of probabilities. If we assume that an image consists of n pixels, we can calculate the joint distribution of an image by calculating the product of conditional probabilities.</p>
  <Latex>{String.raw`p(\mathbf{x}) = p(x_1) * p(x_2 | x_1) * p(x_2 | x_1, x_2) * \cdots * p(x_n | x_1, \cdots, x_{n-1})`}</Latex>  
  <p>We can also express the same idea using the more convenient product notation.</p>
  <Latex>{String.raw`p(\mathbf{x}) = \prod_i^n p(x_i | x_1, \cdots x_{i-1})`}</Latex>  
  <p>Learning the conditional distribution is often much easier, that learning the joint distribution. Imagine you have been provided with a half finished image, filling in the blanks should be relatively straightforward. Even though you do not know how the image looks like exactly, you can manage this task quite easily. Even if you are not an artist, if we give you an image, where only the last pixel is missing, you will be able to fill in the blanks. If on the other hand you need to draw a new image from scratch, you will have a much harder time.</p>
  <p>In the following sections we will study and implement two related autoregressive generative models: PixelRNN and PixelCNN.</p>
</Container>

<div class="separator"></div>
