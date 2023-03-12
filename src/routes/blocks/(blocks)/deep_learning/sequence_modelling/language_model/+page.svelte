<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Border from "$lib/diagram/Border.svelte";
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Language Model</title>
  <meta
    name="description"
    content="A language model assigns a probability to a sequence of words. In deep learning we try to find the sentence with the highest possible probability."
  />
</svelte:head>

<h1>Language Model</h1>
<div class="separator" />

<Container>
  <Alert type="danger"
    >This section is in the process of being refactored.</Alert
  >
  <p>
    So far we have discussed different types of recurrent neural networks that
    can produce text, like the encoder-decoder architecture that is useful to
    translate sentences from one language into a different language. But we have
    not yet discussed how we can actually produce a word or a whole sentence.
  </p>
  <p>
    To understand how we can generate text, we need to discuss what a language
    model is.
  </p>
  <Alert type="info">
    A language model calculates the probability P for a sequence of words <Latex
      >w_1, w_2, ..., w_T</Latex
    >: <Latex>P(w_1, w_2, ..., w_T)</Latex>
  </Alert>
  <p>
    In a translation task we do not create the sentence from thin air, but
    condition the translation on the original language sentence, <Latex
      >P(w_1, w_2, ..., w_T | o_1, o_2, ... , o_T)</Latex
    >.
  </p>
  <p>
    Generating the whole translation in a single go is infeasable, but we can
    transform that model into one that generates one word at a time: using the
    so called chain rule of probability.
  </p>
  <Alert type="info">
    The chain rule of prabability, also called the product rule, rearanges the
    conditional probability <Latex
      >{String.raw`P(A | B) = \dfrac{P(A,B)}{B}`}</Latex
    > into the product form <Latex
      >{String.raw`P(A,B) = P(B \mid A) \cdot P(A)`}</Latex
    >
  </Alert>
  <p>
    If we apply this rule to a two word sentence we end up with the following: <Latex
      >{String.raw`P(w_1, w_2|o_1, ..., o_T) = P(w_1 | o_1, ..., o_t)P(w_2 | w_1, o_1, ..., o_t)`}</Latex
    >
  </p>
  <p>
    This allows us to generate one word at a time and to condition the
    probability of the next word on the previously generated words.
  </p>
  <Latex
    >{String.raw`P(w_1, ..., w_T) = \prod_i^T P(w_i | w_1, ..., w_{i-1}, o_1, ..., o_T)`}</Latex
  >
  <p>
    Our overall goal is to generate a sentence (a translation) that maximizes
    this probability.
  </p>
  <p><Latex>{String.raw`\arg \max_{w_1,...,w_T} P(w_1, ..., w_T)`}</Latex></p>
  <p>
    So how does this work in practice with a recurrent neural network? Let's
    work through a translation example with a encoder-decoder rnn network.
  </p>

  <SvgContainer maxWidth={"1000px"}>
    <svg viewBox="0 0 650 160">
      {#each Array(5) as _, idx}
        <g transform="translate({idx * 120 - 20}, 0)">
          {#if idx < 2}
            <Arrow
              strokeWidth="2"
              data={[
                { x: 50, y: 140 },
                { x: 50, y: 100 },
              ]}
            />
            <Block
              text="X {idx + 1}"
              fontSize={12}
              x="50"
              y="150"
              width="25"
              height="15"
              color="var(--main-color-4)"
            />
          {/if}
          {#if idx >= 2}
            <Arrow
              strokeWidth="2"
              data={[
                { x: 50, y: 55 },
                { x: 50, y: 10 },
              ]}
            />
            <Arrow
              strokeWidth="2"
              dashed={true}
              strokeDashArray="4 4"
              data={[
                { x: 65, y: 20 },
                { x: 110, y: 105 },
                { x: 170, y: 105 },
                { x: 170, y: 98 },
              ]}
            />
            <Block
              text="Y_{idx + 1 - 2}"
              fontSize={12}
              x="65"
              y="20"
              width="25"
              height="15"
              color="var(--main-color-4)"
            />
          {/if}
          <Arrow
            strokeWidth="2"
            data={[
              { x: 70, y: 75 },
              { x: 140, y: 75 },
            ]}
          />
          <Block
            x="50"
            y="75"
            width="30"
            height="30"
            color="var(--main-color-3)"
          />
          <Block
            text="H_{idx + 1}"
            fontSize={12}
            x="125"
            y="65"
            width="25"
            height="15"
            color="var(--main-color-4)"
          />
        </g>
      {/each}
      <Block
        x="100"
        y="22"
        width="100"
        height="25"
        text="Encoder"
        color="none"
        fontSize={20}
      />
      <Block
        x="500"
        y="140"
        width="100"
        height="25"
        text="Decoder"
        color="none"
        fontSize={20}
      />
      <Border x="10" y="50" width="180" height="110" />
      <Border x="250" y="1" width="370" height="110" />
    </svg>
  </SvgContainer>
  <p>
    As previously discussed the encoder compresses the sentence in the original
    language into a vecor representation, H_2 in the image above, and passes it
    to the decoder. The decoder produces one sentence at a time based on the
    hidden state H and the previously generated word Y. To generate the output Y
    the hidden state is used as input into a linear layer that generates as many
    outputs (logits) as there are words in the target language corpus. When we
    use these logits as an input into a softmax function, we generate a
    probability distribution, from which we can draw a sample.
  </p>
  <p>
    There are many ways to generate samples from a distribution, we are going to
    use the one that is easiest to impement, called <Highlight
      >greedy search</Highlight
    >. Greedy search always samples the word with the highest probability
    conditioned on the previously produced word Y and the hidden state H. This
    approach is not optimal, because we are searching for the sentence with the
    highest probability <Latex>{String.raw`P(w_1, ..., w_T)`}</Latex>. Greedy
    search on the other hand looks at the highest probability of the next word
    and not the overall sentence. If we wanted to compare probabilities of full
    sentences, we would have to create an unlimited amount of sentences. Greedy
    search is good enough in our case. If you ever want to produce sentences
    with higher probabilities we recommend you look into <Highlight
      >beam search</Highlight
    >. Beam search doesnt't create a single greedy sample, but generates n most
    likely words at each step. That allows you to compare and contrast those
    paths and to pick the one with the highest overall probability.
  </p>
  <div class="separator" />
</Container>
