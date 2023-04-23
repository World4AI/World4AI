<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
      author: "Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever",
      title: "Improving Language Understanding by Generative Pre-Training",
      journal: "",
      year: "2018",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author:
        "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever",
      title: "Language Models are Unsupervised Multitask Learners",
      journal: "",
      year: "2019",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author:
        "Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et. al",
      title: "Language Models are Few-Shot Learners",
      journal: "",
      year: "2020",
      pages: "",
      volume: "",
      issue: "",
    },
  ];

  const sentence = ["what", "is", "your", "name"];
</script>

<svelte:head>
  <title>GPT - World4AI</title>
  <meta
    name="description"
    content="GPT, short for generative pre-training, is a family of decoder based transformer models developed by OpenAI. The models are trained based on next token predictions and can perform zero-shot, one-shot or few-shots learning."
  />
</svelte:head>

<Container>
  <h1>GPT</h1>
  <div class="separator" />
  <p>
    <Highlight>GPT</Highlight>, short for generative pre-training , is a family
    of models developed by researchers at OpenAI. GPT is a decoder based
    transformer model, without any encoder interaction. We simply stack layers
    of decoders on top of each other.
  </p>
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 300">
      <Arrow
        data={[
          { x: 100, y: 310 },
          { x: 100, y: 6 },
        ]}
        strokeWidth={2}
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
          class="fill-blue-100"
          text="Decoder Layer"
        />
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Due to the lack of encoders in the GPT architecture we do not need any
    cross-attention. This simplifies the decoder layer to just two sublayers:
    masked multihead attention and position-wise feed-forward neural network.
  </p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 260 360">
      <Block x={150} y={175} width={200} height={300} text={""} fontSize={20} />
      <!-- 3 bottom arrows -->
      <Arrow
        data={[
          { x: 150, y: 350 },
          { x: 150, y: 275 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />
      <Arrow
        data={[
          { x: 150, y: 350 },
          { x: 150, y: 290 },
          { x: 90, y: 290 },
          { x: 90, y: 275 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />
      <Arrow
        data={[
          { x: 150, y: 350 },
          { x: 150, y: 290 },
          { x: 210, y: 290 },
          { x: 210, y: 275 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />
      <!-- first skip connection -->
      <Arrow
        data={[
          { x: 150, y: 350 },
          { x: 150, y: 310 },
          { x: 10, y: 310 },
          { x: 10, y: 200 },
          { x: 65, y: 200 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />

      <!-- connect attention to feed forward -->
      <Arrow
        data={[
          { x: 150, y: 250 },
          { x: 150, y: 125 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />

      <!-- second skip connection -->
      <Arrow
        data={[
          { x: 150, y: 170 },
          { x: 10, y: 170 },
          { x: 10, y: 50 },
          { x: 65, y: 50 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />
      <!-- connect fc to next layer-->
      <Arrow
        data={[
          { x: 150, y: 100 },
          { x: 150, y: 10 },
        ]}
        strokeWidth={2}
        strokeDashArray="4 4"
        dashed={true}
      />
      <!-- decoder components -->
      <Block
        x={150}
        y={50}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={150}
        y={100}
        width={150}
        height={30}
        text={"P.w. Feed Forward"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={150}
        y={200}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={150}
        y={250}
        width={150}
        height={30}
        text={"Multihead Atention"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={43}
        y={250}
        width={60}
        height={20}
        text={"Masked"}
        fontSize={10}
        class="fill-red-400"
      />
    </svg>
  </SvgContainer>
  <p>
    The training objective is of GPT is quite simple. Given some tokens from a
    sentence, predict the next token. For example if the GPT is given the three
    words <em>what is your</em>, the stack of decores should predict words like
    <em>name</em>, <em>age</em> or <em>weight</em>.
  </p>
  <SvgContainer maxWidth={"400px"}>
    <svg viewBox="0 0 200 40">
      {#each sentence as word, idx}
        <Block
          x={25 + idx * 50}
          y={20}
          width={40}
          height={20}
          text={word}
          fontSize={10}
          class={idx === 3 ? "fill-black" : "fill-lime-100"}
        />
      {/each}
    </svg>
  </SvgContainer>
  <p>
    Unline BERT, GPT is unidirectional and we need to maks out future words. The
    tokens must never pay attention to future tokens, as that would contaminate
    the training process by allowing the tokens to attend to tokens that they
    are actually trying to predict. So if we want to predict the fourth token
    based on the third embedding that was processed by layers of decoders, the
    third embedding can attend to all previous tokens, including itself, but not
    the token it tries to predict.
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
  <p>
    Based on the pre-trainig task, it is obvious that GPT can be used for text
    generation. You provide the model with a starting text and the model
    generates word after word to finish your writing, using the previously
    generated words as input in a recursive manner.
  </p>
  <p>
    GPT is not a single model, but a family of models. By now we have GPT-1<InternalLink
      id={1}
      type="reference"
    />, GPT-2<InternalLink id={2} type="reference" />, GPT-3<InternalLink
      id={3}
      type="reference"
    /> and GPT-4. With each new iteration OpenAI increased the size of the models
    and the datasets that the models were trained on. It became clear that you could
    scale up transformer-like models through size and data and the performance would
    improve. Unfortunately for the deep-learning community OpenAI decided starting
    with GPT-2 not to release the pre-trained models to the public due to security
    concerns and profit considerations. While the weights of GPT-2 were released
    eventually, the public has no direct access to GPT-3 or GPT-4. Only throught
    the OpenAI api can you interract with newest GPT models. Luckily there are companies,
    like
    <a href="https://www.eleuther.ai/" target="_blank" rel="noreferrer"
      >EleutherAI</a
    >, that attemt to replicate the OpenAI GPT-3/GPT-4 models. At the moment of
    writing their largest model, GPT-NeoX-20B, consists of 20 billion
    parameters, but they plan to train even larger models to match the
    performance of the newest models by OpenAI.
  </p>
  <p>
    We can use the transformers library by HuggingFace to interract with GPT-2.
    The easiest way to accomplish that is to use the text-generation pipeline. A
    pipeline abstracts away most of code running in the background. We do not
    need to take care of the tokenizer or the model, just by using
    'text-generation' as the input, HuggingFace downloads GPT-2 weights and
    allows you to generate text.
  </p>
  <PythonCode
    code={`from transformers import pipeline

generator = pipeline("text-generation")
prompt = (
    "In the year 2035 humanity will have created human level artificial intelligence."
)
outputs = generator(prompt, max_length=100)`}
  />
  <p>
    When we use the prompt <em>
      "In the year 2035 humanity will have created human level artificial
      intelligence."
    </em>
    we might get the following result.
  </p>
  <p class="font-mono bg-slate-50 p-3">
    In the year 2035 humanity will have created human level artificial
    intelligence. Some experts believe that the first phase of AI could arrive
    around 2030 and be capable of being applied in a multitude of applications,
    including transportation, health or the arts.\n\nAs humans live longer and
    more efficiently and interact more quickly with computers, we will probably
    see a gradual step towards being a multi-systemed society. A society which
    will have more people involved and will focus on information management,
    education, marketing, governance,
  </p>
  <p>
    This is relatively cohesive, but the results will depend on your initial
    prompt and will change each time you run the code.
  </p>
</Container>
<Footer {references} />
