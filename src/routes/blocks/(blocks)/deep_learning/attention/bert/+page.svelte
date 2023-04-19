<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  // imports for the diagrams
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const tokens = [
    "[CLS]",
    "[MASK]",
    "TOK 2",
    "[SEP]",
    "TOK 1",
    "TOK 2",
    "[SEP]",
  ];

  const clsTokens = [
    "[CLS]",
    "TOK 1",
    "TOK 2",
    "TOK 3",
    "TOK 4",
    "TOK 5",
    "[SEP]",
  ];

  const references = [
    {
      author:
        "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina",
      title:
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      year: "2018",
    },
  ];
</script>

<svelte:head>
  <title>BERT - World4AI</title>
  <meta
    name="description"
    content="BERT (short for biderectional encoder representation from transformers) is a pretrained language model, that can used for different fine-tuning tasks, like text summarization, sentiment analysis and much more. BERT (and its relatives) has become the de facto standard transfer learning tool for natural language processing."
  />
</svelte:head>

<Container>
  <h1>BERT</h1>
  <div class="separator" />
  <p>
    When the original transformer paper was released in the year 2017, it was
    not clear what tremendous impact that architecture would have on deep
    learning. Slowly but surely researchers from different research labs begun
    to release transformer-based architectures and the more time passed, the
    more areas were conquered by transformers. One of the first models that
    garnered a lot of attention was <Highlight>BERT</Highlight> from Google<InternalLink
      id={1}
      type="reference"
    />. BERT is a pre-trained language model that allowes practitioners to
    fine-tune the model to their specific needs. Nowadays BERT (and its
    relatives) is de facto standard tool that is used for transfer learning in
    the area of natural language processing.
  </p>
  <div class="separator" />

  <h2>Architecture</h2>
  <p>
    BERT is short for <Highlight>B</Highlight>iderectional <Highlight
      >E</Highlight
    >ncoder <Highlight>R</Highlight>epresentation from <Highlight>T</Highlight
    >ransformers. We can infer from the name, that the model architecture
    consists solely from a stack of transformer encoders, without a decoder.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 200 300">
      <Arrow
        data={[
          { x: 100, y: 300 },
          { x: 100, y: 5 },
        ]}
        strokeWidth={1.6}
        dashed={true}
        strokeDashArray="4 4"
        moving={true}
      />
      {#each Array(6) as _, idx}
        <Block
          x={100}
          y={30 + idx * 50}
          width={160}
          height={30}
          fontSize={15}
          class="fill-green-100"
          text="Encoder Layer"
        />
      {/each}
    </svg>
  </SvgContainer>
  <p>
    The original BERT paper introduced two models. The <Highlight
      >BERT-Base</Highlight
    > model consists of 12 encoder layers, each with 12 attention heads and 768 hidden
    units for each token. <Highlight>BERT-Large</Highlight> on the other hand uses
    24 layers with 16 heads and 1024 hidden units.
  </p>
  <p>
    Each layer takes a certain amount of tokens and outputs the same number of
    tokens. The number of tokens never changes between the encoder layers.
  </p>
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 200 150">
      {#each Array(5) as _, idx}
        <Block
          x={30 + idx * 35}
          y={10}
          width={10}
          height={10}
          class="fill-yellow-100"
        />
        <Arrow
          data={[
            { x: 30 + idx * 35, y: 55 },
            { x: 30 + idx * 35, y: 25 },
          ]}
          strokeWidth={1.5}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
        />
        <Block
          x={30 + idx * 35}
          y={140}
          width={10}
          height={10}
          class="fill-red-300"
        />
        <Arrow
          data={[
            { x: 30 + idx * 35, y: 130 },
            { x: 30 + idx * 35, y: 100 },
          ]}
          strokeWidth={1.5}
          dashed={true}
          strokeDashArray="4 4"
          moving={true}
        />
      {/each}
      <Block
        x={100}
        y={75}
        width={160}
        height={30}
        fontSize={15}
        class="fill-green-100"
        text="Encoder Layer"
      />
    </svg>
  </SvgContainer>
  <p>
    The <em>biderectional</em> part means, that the encoder can pay attention to
    all tokens contained in the sequence. For that reason BERT is primarily used
    for tasks, that have access to the full sequence at inference time. For example
    in a classification task we process the whole sequence in order to classify the
    sentence.
  </p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 300 150">
      <Block x={40} y={10} width={50} height={12} text="BERT Family" />
      {#each Array(6) as _, idx}
        <Block
          x={25 + idx * 50}
          y={130}
          width={20}
          height={20}
          text="T_{idx + 1}"
          class="fill-red-300"
        />
        <Block
          x={25 + idx * 50}
          y={35}
          width={20}
          height={20}
          text="T_{idx + 1}"
          class="fill-yellow-100"
        />
        {#each Array(6) as _, idx2}
          <Arrow
            data={[
              { x: 25 + idx * 50, y: 120 },
              { x: 25 + idx2 * 50, y: 45 },
            ]}
            showMarker={false}
            dashed={true}
            moving={true}
            speed={50}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    In the next section we will additionally encouter the GPT family of models,
    that are created by stacking transformer decoders. A GPT like decoder
    creates output tokens, based on all current or previous input
    embeddings/tokens.
  </p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 300 150">
      <Block x={40} y={10} width={50} height={12} text="GPT Family" />
      {#each Array(6) as _, idx}
        <Block
          x={25 + idx * 50}
          y={130}
          width={20}
          height={20}
          text="T_{idx + 1}"
          class="fill-red-300"
        />
        <Block
          x={25 + idx * 50}
          y={35}
          width={20}
          height={20}
          text="T_{idx + 1}"
          class="fill-yellow-100"
        />
        {#each Array(6) as _, idx2}
          {#if idx2 >= idx}
            <Arrow
              data={[
                { x: 25 + idx * 50, y: 120 },
                { x: 25 + idx2 * 50, y: 45 },
              ]}
              showMarker={false}
              dashed={true}
              moving={true}
              speed={50}
            />
          {/if}
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <div class="separator" />

  <h2>Pre-Training</h2>
  <p>
    BERT was pre-trained jointly on two different objectives: <Highlight
      >masked language model</Highlight
    > and <Highlight>next section prediction</Highlight>.
  </p>
  <p>
    In a masked language model, we replace some parts of the original sequence
    randomly with a special [MASK] token and the model has to predict the
    missing word. Let's for example look at the below sentence to understand,
    why this task matters for pretraining.
  </p>
  <p class="text-center">
    today i went to a <span
      class="inline-block py-1 px-2 bg-lime-100 font-bold border border-black"
      >[MASK]</span
    > to get my hair done
  </p>
  <p>
    While you probably have a couple of options to fill out the masked word,
    your options are still limited by logic and the rules of the english
    language. The word "hairdresser" is probably the most likely option, but the
    word "salon" or even "friend" are also valid options. When a model learns to
    replace [MASK] by a valid word, that shows that in the very least, the model
    learned the basic statistics that govern the english language. And those
    statistics are useful not only for the task at hand, but also for many other
    tasks that require natural language understanding.
  </p>
  <p>
    In the next section prediction task, the model faces two sequences and has
    to predict if the second sequence is the logical continuation of the first
    sequence.
  </p>
  <p>
    The two sentences below seem to have a logical connection, so we expect the
    model to return true.
  </p>
  <div class="text-center font-bold mb-4">
    <span class="inline-text py-1 px-2 bg-lime-100 border border-black"
      >a man went to the store</span
    >
  </div>
  <div class="text-center font-bold">
    <span class="inline-text py-1 px-2 bg-lime-100 border border-black"
      >he bought a gallon of milk</span
    >
  </div>
  <p>
    The next two sentences on the other hand are unrelated and the model should
    return false.
  </p>
  <div class="text-center font-bold mb-4">
    <span class="inline-text py-1 px-2 bg-lime-100 border border-black"
      >a man went to the store</span
    >
  </div>
  <div class="text-center font-bold">
    <span class="inline-text py-1 px-2 bg-red-100 border border-black"
      >penguins are flightless birds</span
    >
  </div>
  <p>
    Similarly to the masked model, solving this task is related to understanding
    the english language. Once a model is competent at solving this task, we can
    assume that it has learned some important statistics of the language and
    those statistics can be used for downstream tasks.
  </p>
  <p>
    Let's for a second assume, that we are training on two sentences, each
    consisting of 2 words (2 tokens).
  </p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 350 100">
      {#each tokens as token, idx}
        <Block
          x={22 + idx * 51}
          y={90}
          width={38}
          height={16}
          text={token}
          fontSize={10}
          class="fill-red-300"
        />
        <Block
          x={22 + idx * 51}
          y={10}
          width={38}
          height={16}
          text={token}
          fontSize={10}
          class="fill-yellow-100"
        />
        <Arrow
          data={[
            { x: 22 + idx * 51, y: 80 },
            { x: 22 + idx * 51, y: 25 },
          ]}
          strokeWidth={1.5}
        />
      {/each}
      <Block
        x={175}
        y={50}
        width={345}
        height={20}
        text={"Encoder Layers"}
        class="fill-green-100"
        fontSize={12}
      />
    </svg>
  </SvgContainer>
  <p>
    We first prepend the two sentences with the [CLS] token. Once this token is
    processed by a stack of encoders, it is used for the binary classification
    to determine if the second sentence should follow the first sentence: the
    next section prediction task. We separate the two sentences by the [SEP]
    token and additionally append this token at the end of the sentence. We mask
    out some of the words for the masked language model. Those masked tokens are
    also processed by layers of encoders and are used to predict the correct
    token that was masked out. Both losses are aggregated for the gradient
    descent step.
  </p>
  <p>
    It is important to mention that both those tasks are trained using
    self-supervised learning. We do not require anyone to collect and label a
    dataset. We can for example use Wikipedia articles and pick out random
    consecutive sentences and mask out some of the tokens. This gives us a huge
    dataset for pre-training.
  </p>
  <div class="separator" />

  <h2>Fine-Tuning</h2>
  <p>
    When we have a labeled language dataset, we can use BERT for fine-tuning.
    BERT can be used How the pre-trained BERT model can be used for fine-tuning
    depends on the task at hand.
  </p>
  <p>
    Let's assume we are dealing with a classification task, like sentiment
    analysis. The first token, CLS, is the start token and is generally used for
    classification tasks. We can use the embedding for the class token from the
    last encoder layer as an input into a classification layer and ignore the
    rest.
  </p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 350 100">
      {#each clsTokens as token, idx}
        <Block
          x={22 + idx * 51}
          y={90}
          width={38}
          height={16}
          text={token}
          fontSize={10}
          class="fill-red-300"
        />
        <Block
          x={22 + idx * 51}
          y={10}
          width={38}
          height={16}
          text={token}
          fontSize={10}
          class={idx === 0 ? "fill-yellow-100" : "fill-black"}
        />
        <Arrow
          data={[
            { x: 22 + idx * 51, y: 80 },
            { x: 22 + idx * 51, y: 25 },
          ]}
          strokeWidth={1.5}
        />
      {/each}
      <Block
        x={175}
        y={50}
        width={345}
        height={20}
        text={"Encoder Layers"}
        class="fill-green-100"
        fontSize={12}
      />
    </svg>
  </SvgContainer>
  <p>
    BERT is designed for fine-tuning, so it makes no sense to train the model
    from scratch. Instead we will use the pre-trained BERT weights to solve our
    task at hand. Nowadays the most efficient way to use BERT is with the help
    of the <a href="https://huggingface.co/" target="_blank" rel="noreferrer"
      >🤗HuggingFace library</a
    >. This library contains models, pretrained weights, datasets and much more.
    We will make heavy use of it in future, and not only for natural language
    processing.
  </p>
</Container>
<Footer {references} />
