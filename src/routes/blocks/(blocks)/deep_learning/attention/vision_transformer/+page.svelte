<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  // imports for the diagrams
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
      author:
        "Alexander Kolesnikov and Alexey Dosovitskiy and Dirk Weissenborn and Georg Heigold and Jakob Uszkoreit and Lucas Beyer and Matthias Minderer and Mostafa Dehghani and Neil Houlsby and Sylvain Gelly and Thomas Unterthiner and Xiaohua Zhai",
      title:
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
      journal: "",
      year: "2021",
      pages: "",
      volume: "",
      issue: "",
    },
  ];

  const imgWidth = 4;
  const imgHeight = 4;
  const pixelSize = 20;
  const patchSize = 2;

  const colors = [
    "fill-red-100",
    "fill-blue-100",
    "fill-yellow-100",
    "fill-green-100",
  ];

  const colorsProjected = [
    "fill-white",
    "fill-red-100",
    "fill-blue-100",
    "fill-yellow-100",
    "fill-green-100",
  ];
</script>

<svelte:head>
  <title>Vision Transformer - World4AI</title>
  <meta
    name="description"
    content="A vision transformer is an encoder based computer vision architecture, that processes patches of pixels in parallel."
  />
</svelte:head>

<Container>
  <h1>Vision Transformer</h1>
  <div class="separator" />
  <p>
    We have mentioned before that the transformer has become the swiss army
    knive for the deep learning community. This transformer has a general
    purpose architecture that can be applied to many modalities. In this section
    we will discuss how we can use layers of transformer encoders for computer
    vision, the so called <Highlight>vision transformer</Highlight><InternalLink
      id={1}
      type="reference"
    />.
  </p>
  <p>
    For the sake of explanation let's assume that we are dealing with images of
    size 4x4 pixels.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 100 100">
      {#each Array(imgWidth) as _, wIdx}
        {#each Array(imgHeight) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + wIdx * (pixelSize + 2)}
            y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class="fill-gray-200"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    The naive approach would be to allow each pixels to attend to each other
    pixel in the image, but the computational cost for the calculation of
    attention would explode when the size of the image increases. Instead we
    divide the image into quadratic patches. For our small example we could
    create patches of size 2x2.
  </p>
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 100 100">
      {#each Array(imgWidth) as _, wIdx}
        {#each Array(imgHeight) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + wIdx * (pixelSize + 2)}
            y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colors[
              Math.floor(wIdx / patchSize) +
                (Math.floor(hIdx / patchSize) * imgWidth) / patchSize
            ]}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Remember that the transformer expects vector embeddings as inputs. To
    achieve that we first flatten the patches.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 200 100">
      {#each Array((imgWidth * imgHeight) / (patchSize * patchSize)) as _, idx}
        {#each Array(patchSize * patchSize) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + idx * (pixelSize + 37)}
            y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colors[idx]}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Next we project the flattened patches linearly, by running each individual
    vector through the same linear layer using the same weights and bias. This
    result can be compared to the token embeddings in the traditional
    transformer. We additionally create a separate learnable embedding (vector
    on the left in the image below). This embedding corresponds to the
    classification token and will be used at a later stage to classify our
    images.
  </p>
  <SvgContainer maxWidth={"400px"}>
    <svg viewBox="0 0 260 250">
      {#each Array((imgWidth * imgHeight) / (patchSize * patchSize)) as _, idx}
        {#each Array(patchSize * patchSize) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + (idx + 1) * (pixelSize + 37)}
            y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colors[idx]}
          />
        {/each}
      {/each}
      {#each Array((imgWidth * imgHeight) / (patchSize * patchSize)) as _, idx}
        <Arrow
          data={[
            {
              x: 5 + pixelSize / 2 + (idx + 1) * (pixelSize + 37),
              y: 140 + pixelSize / 2,
            },
            {
              x: 5 + pixelSize / 2 + (idx + 1) * (pixelSize + 37),
              y: 90 + pixelSize / 2,
            },
          ]}
          strokeWidth={1.5}
          strokeDashArray="4 4"
          dashed={true}
          moving={true}
        />
        {#each Array(patchSize * patchSize) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + (idx + 1) * (pixelSize + 37)}
            y={155 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colors[idx]}
          />
        {/each}
      {/each}
      <Block
        x={160}
        y={120}
        width={190}
        height={pixelSize}
        text="Linear Projection"
        fontSize={12}
        class="fill-white"
      />

      {#each Array(patchSize * patchSize) as _, hIdx}
        <Block
          x={5 + pixelSize / 2}
          y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
          width={pixelSize}
          height={pixelSize}
          class="fill-white"
        />
      {/each}
    </svg>
  </SvgContainer>
  <p>
    At this point our model can not differentiate between the different
    positions of the vectors in the image. Therefore we create positional
    embeddings and add those to linear projections.
  </p>
  <SvgContainer maxWidth={"400px"}>
    <svg viewBox="0 0 260 250">
      {#each Array((imgWidth * imgHeight) / (patchSize * patchSize) + 1) as _, idx}
        {#each Array(patchSize * patchSize) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + idx * (pixelSize + 37)}
            y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colorsProjected[idx]}
          />
        {/each}
      {/each}
      {#each Array((imgWidth * imgHeight) / (patchSize * patchSize) + 1) as _, idx}
        <Arrow
          data={[
            {
              x: 5 + pixelSize / 2 + idx * (pixelSize + 37),
              y: 140 + pixelSize / 2,
            },
            {
              x: 5 + pixelSize / 2 + idx * (pixelSize + 37),
              y: 90 + pixelSize / 2,
            },
          ]}
          strokeWidth={1.5}
          strokeDashArray="4 4"
          dashed={true}
          moving={true}
        />
        {#each Array(patchSize * patchSize) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + idx * (pixelSize + 37)}
            y={155 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colorsProjected[idx]}
          />
        {/each}
      {/each}
      <Block
        x={128}
        y={120}
        width={250}
        height={pixelSize}
        text="Add Positional Embedding"
        fontSize={12}
        class="fill-white"
      />
    </svg>
  </SvgContainer>
  <p>
    Finally we pass those embeddings through layers of encoders and allow each
    embedding to attend to each other embedding. At the output layer we ignore
    all vectors, except the one corresponding to the classification token. This
    token is used as the input to a fully connected layer and we train the whole
    network jointly on a classification task.
  </p>
  <SvgContainer maxWidth={"400px"}>
    <svg viewBox="0 0 260 250">
      {#each Array(patchSize * patchSize) as _, hIdx}
        <Block
          x={5 + pixelSize / 2}
          y={5 + pixelSize / 2 + hIdx * (pixelSize + 2)}
          width={pixelSize}
          height={pixelSize}
          class="fill-white"
        />
      {/each}
      {#each Array((imgWidth * imgHeight) / (patchSize * patchSize) + 1) as _, idx}
        <Arrow
          data={[
            {
              x: 5 + pixelSize / 2 + idx * (pixelSize + 37),
              y: 140 + pixelSize / 2,
            },
            {
              x: 5 + pixelSize / 2 + idx * (pixelSize + 37),
              y: 90 + pixelSize / 2,
            },
          ]}
          strokeWidth={1.5}
          strokeDashArray="4 4"
          dashed={true}
          moving={true}
        />
        {#each Array(patchSize * patchSize) as _, hIdx}
          <Block
            x={5 + pixelSize / 2 + idx * (pixelSize + 37)}
            y={155 + pixelSize / 2 + hIdx * (pixelSize + 2)}
            width={pixelSize}
            height={pixelSize}
            class={colorsProjected[idx]}
          />
        {/each}
      {/each}
      <Block
        x={128}
        y={120}
        width={250}
        height={pixelSize}
        text="Encoder Layers"
        fontSize={12}
        class="fill-white"
      />
    </svg>
  </SvgContainer>
</Container>
<Footer {references} />
