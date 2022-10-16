<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";

  import ButtonContainer from "$lib/button/ButtonContainer.svelte"; 
  import PlayButton from "$lib/button/PlayButton.svelte"; 

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
  <p>We have already encountered autoregressive models, when we were dealing with language models, like GPT. Simply put an autoregressive model relies on its previous values to generate the next value. So if you want to produce the fifth word in a sentence, you provide the model with the previous four words. In this section we will deal mostly with autoregressive generative models to generate images, therefore we will generate one pixel at a time.</p>
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
  <p>The example above shows how pixel generation looks like. To generate the red pixel, the model is allowed to look at all previous (yellow) pixels. Looking ahead is obviously not allowed.</p>
</Container>

<div class="separator"></div>
