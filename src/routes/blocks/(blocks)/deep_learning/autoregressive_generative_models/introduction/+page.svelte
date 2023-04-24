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
    if (activeCol < imageLength - 1) {
      activeCol += 1;
    } else if (activeRow < imageLength - 1) {
      activeCol = 0;
      activeRow += 1;
    } else {
      activeRow = 0;
      activeCol = 0;
    }
  }
</script>

<svelte:head>
  <title>Autoregressive Generative Models - World4AI</title>
  <meta
    name="description"
    content="Autoregressive generative models, like GPT or PixelRNN, generate one element at a time, where the next element depends on all previously generated elements. During the trianing process the model is not allowed to look at the future tokens, therefore autoregressive models usually implement some sort of masking."
  />
</svelte:head>

<Container>
  <h1>Autoregressive Generative Models</h1>
  <div class="separator" />
  <p>
    We have already encountered autoregressive models, when we were dealing with
    decoder-based language models, like GPT. An autoregressive model relies on
    its previously generated values. If for example you want to produce the
    fifth word in a sentence, you provide the model with the previous four
    words.
  </p>
  <p>
    Autogenerative models can also be used to generate images. We simply
    generate one pixel at a time, while conditioning the model on previously
    generated pixels.
  </p>
  <ButtonContainer>
    <PlayButton {f} delta={70} />
  </ButtonContainer>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 300">
      {#each Array(imageLength) as _, colIdx}
        {#each Array(imageLength) as _, rowIdx}
          <rect
            x={2 + colIdx * (pixelSize + 5)}
            y={2 + rowIdx * (pixelSize + 5)}
            width={pixelSize}
            height={pixelSize}
            class={colIdx === activeCol && rowIdx === activeRow
              ? "fill-red-500 stroke-black"
              : rowIdx < activeRow ||
                (rowIdx === activeRow && colIdx < activeCol)
              ? "fill-blue-100 stroke-black"
              : "fill-white"}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    The interactive example above shows how pixel generation looks like. To
    generate the next (red) pixel, the model looks at all previous (blue)
    pixels. As you can imagine this recursive process is quite slow and scales
    quite badly with increased image size. While autoregressive procedures are
    state of the art for text generation, for image generation there are more
    efficient techniques, that we will introduce in future chapters. This
    chapter is merely an intorduction into the world of generative models.
  </p>
  <p>
    The goal of a generative model is to estimate the probability distribution <Latex
      >{String.raw`p(\mathbf{x})`}</Latex
    >.
    <Latex>{String.raw`\mathbf{x}`}</Latex> could by a piece of text, a song or in
    our case a full image. Once we have learned this distrubution, we can draw random
    images from it.
  </p>
  <p>
    For autoregressive generative models we rewrite this task by using the chain
    rule of probabilities and express the probability distribution as the
    product of conditional probabilities.
  </p>
  <Latex
    >{String.raw`p(\mathbf{x}) = p(x_1) * p(x_2 | x_1) * p(x_2 | x_1, x_2) * \cdots * p(x_n | x_1, \cdots, x_{n-1})`}</Latex
  >
  <p>
    Essentially we are answering the following question: <em
      >'What is the probability of the next pixel, given that I have observed
      all those previous pixels?'</em
    >. Learning the conditional distribution is often much easier, than learning
    the joint distribution. Imagine you have been provided with a half finished
    image, filling in the blanks should be relatively straightforward. Even
    though you do not know how the image looks like exactly, you can manage this
    task much better than creating an image from scratch. Even if you are not an
    artist, if we give you an image, where only the last pixel is missing, you
    will be able to fill in the blanks.
  </p>
  <div class="separator" />
</Container>
