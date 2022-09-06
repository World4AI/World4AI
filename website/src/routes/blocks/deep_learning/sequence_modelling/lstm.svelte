<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Slider from "$lib/Slider.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Border from "$lib/diagram/Border.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import Multiply from "$lib/diagram/Multiply.svelte";
  import Circle from "$lib/diagram/Circle.svelte";

  const references = [
    {
      author: "S. Hochreiter and J. Schmidhuber",
      title: "Long Short-Term Memory",
      journal: "Neural Computation,",
      year: "1997",
      pages: "1735-1780",
      volume: "9",
      issue: "8",
    },
  ];

  let dummyGateState = 1;
</script>

<svelte:head>
  <title>World4AI | Deep Learning | LSTM</title>
  <meta
    name="description"
    content="The LSTM (long short-term memory) cell uses a gated architecture in order to learn and preserve long-term dependencies. Unlike the simple RNN neural network, LSTM networks can deal with large sequences."
  />
</svelte:head>

<h1>LSTM</h1>
<div class="separator" />

<Container>
  <p>
    Often a recurrent neural network has to process sequences with 100 or more
    steps.
  </p>
</Container>
<SvgContainer maxWidth={"1500px"}>
  <svg viewBox="0 0 800 150">
    {#each Array(3) as _, idx}
      <g transform="translate({idx * 120 - 20}, 0)">
        <Arrow
          strokeWidth="1"
          data={[
            { x: 50, y: 140 },
            { x: 50, y: 100 },
          ]}
        />
        <Arrow
          strokeWidth="1"
          data={[
            { x: 50, y: 55 },
            { x: 50, y: 10 },
          ]}
        />
        <Arrow
          strokeWidth="1"
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
          text="X_{idx + 1}"
          fontSize={12}
          x="50"
          y="140"
          width="25"
          height="15"
          color="var(--main-color-4)"
        />
        <Block
          text="Y_{idx + 1}"
          fontSize={12}
          x="65"
          y="20"
          width="25"
          height="15"
          color="var(--main-color-4)"
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
      text={"......."}
      fontSize="12"
      x="400"
      y="75"
      width="30"
      height="10"
      color="var(--main-color-1)"
    />
    {#each Array(3) as _, idx}
      <g transform="translate({idx * 120 + 400}, 0)">
        <Arrow
          strokeWidth="1"
          data={[
            { x: 50, y: 140 },
            { x: 50, y: 100 },
          ]}
        />
        <Arrow
          strokeWidth="1"
          data={[
            { x: 50, y: 55 },
            { x: 50, y: 10 },
          ]}
        />
        <Arrow
          strokeWidth="1"
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
          text="X_{idx + 98}"
          fontSize={12}
          x="50"
          y="140"
          width="40"
          height="15"
          color="var(--main-color-4)"
        />
        <Block
          text="Y_{idx + 98}"
          fontSize={12}
          x="75"
          y="20"
          width="40"
          height="15"
          color="var(--main-color-4)"
        />
        <Block
          text="H_{idx + 98}"
          fontSize={12}
          x="125"
          y="65"
          width="40"
          height="15"
          color="var(--main-color-4)"
        />
      </g>
    {/each}
  </svg>
</SvgContainer>
<Container>
  <p>
    With each time step we incorporate more and more information into the hidden
    state while at same time the distance to the start of the sequence
    increases. Let's take the following language task as an example.
  </p>
  <div class="light-blue">
    <p>
      <Highlight>Computers</Highlight> have been a passion of mine from the very
      early age. I was fascinated by the idea that I could create anything out of
      nothing using only my skills and my imagination. So when the time came to select
      my major, I did not hesitate and picked <Highlight>-----------</Highlight
      >.
    </p>
  </div>
  <p>
    In this task we need to predict the last word(s). The key to solving that
    task is to realize that the very first word in the very first sentence
    provides the necessary context. If that person has been interested in
    computers for such a long time, it is reasonable to assume that <em
      >computer science</em
    >
    or <em>programming</em> would be good choces to fill the blank. Yet the
    distance between the first word and the prediction is roughly 50 words. Each
    time the recurrent neural network processes a piece of the sentence, it
    adjusts the hidden state, so that by the end of the sentence hardly any
    information remains from the beginning of the sentence. The network forgets
    the beginning of the sentence long before it is done reading. A recurrent
    neural network struggles with so called <Highlight
      >long term dependencies</Highlight
    >.
  </p>
  <p>
    Theoretically speaking there is nothing that prevents a recurrent neural
    network from learning such dependencies. Assuming you had the perfect
    weights for a particular task, an RNN should have the capacity to model long
    term dependencies. It is the learning part that makes recurrent networks
    less attractive.
  </p>

  <p>
    Let's try to understand why that is the case. If we assume that our input is
    a scalar <Latex>x</Latex> that equals to 1 and the weight is a scalar <Latex
      >w</Latex
    >. Given that the sequence is of length 100, the output, based on that
    initial input would equal to <Latex>{String.raw`x_1w^{100}`}</Latex>. We
    know that this is a somewhat of an oversimplification, because we ignore
    other parts of the sequence, but the intuition is still valid. If the the
    weight is above 1, then the output will explode, if the weigth is below 1,
    the output will vanish. As the weight of the neural network is used to
    determine the gradients, the exploding and vanishing values will lead to
    exploding and vanishing gradients.
  </p>
  <p>
    Because we use the same weights over and over again, the values and thereby
    the gradients grow or vanish exponentially. Below for example we use a
    factor of 1.2 to demonstrate how the volume of box changes if it is
    constantly multiplied or divided by 1.2.
  </p>
  <svg viewBox="0 0 700 100">
    {#each Array(20) as _, idx}
      <Block
        color="var(--main-color-2)"
        x={idx * 30 + 1.2 ** idx}
        y="25"
        width={1.2 ** idx}
        height={1.2 ** idx}
      />
      <Block
        color="var(--main-color-1)"
        x={idx * 30 + 1.2 ** idx + 10 * idx + 25}
        y="75"
        width={1.2 ** (20 - idx)}
        height={1.2 ** (20 - idx)}
      />
    {/each}
  </svg>
  <p>
    We can use gradient clipping to deal with exploding gradients. In order to
    deal with vanishing gradients we will use a new type of a recurrent neural
    network called long short-term memory, or <Highlight>LSTM</Highlight> for short.
  </p>
  <p>
    The long short-term memory cell was developed by Sepp Hochreiter and Jürgen
    Schmidhuber in the year 1997<InternalLink type="reference" id={1} />. As you
    can imagine this paper was way ahead its time, as it was published during
    the last AI winter, but after 2012, the architecture gained a lot of
    popularity.
  </p>

  <p>
    For the most part we do not change the overarching design of a recurrent
    neural network. The cell produces some output, that is used as an input in
    the next iteration, but we replace a simple RNN cell by a LSTM cell. Unlike
    the regular cell, LSTM cells produce two values: the short term memory <Latex
      >h_t</Latex
    > (hidden value) and the long term memory <Latex>c_t</Latex> (cell value).
  </p>
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 150">
      {#each Array(4) as _, idx}
        <g transform="translate({idx * 120 - 20}, 0)">
          <Arrow
            strokeWidth="1"
            data={[
              { x: 50, y: 140 },
              { x: 50, y: 100 },
            ]}
          />
          <Arrow
            strokeWidth="1"
            data={[
              { x: 50, y: 55 },
              { x: 50, y: 10 },
            ]}
          />
          <Arrow
            strokeWidth="1"
            data={[
              { x: 70, y: 65 },
              { x: 140, y: 65 },
            ]}
          />
          <Arrow
            strokeWidth="1"
            data={[
              { x: 70, y: 85 },
              { x: 140, y: 85 },
            ]}
          />
          <Block
            text={"LSTM"}
            fontSize="10"
            x="50"
            y="75"
            width="30"
            height="30"
            color="var(--main-color-3)"
          />
          <Block
            text={idx + 1}
            fontSize={12}
            x="50"
            y="140"
            width="15"
            height="15"
            color="var(--main-color-4)"
          />
          <Block
            text="C_{idx + 1}"
            fontSize={12}
            x="125"
            y="55"
            width="25"
            height="15"
            color="var(--main-color-4)"
          />
          <Block
            text="H_{idx + 1}"
            fontSize={12}
            x="125"
            y="95"
            width="25"
            height="15"
            color="var(--main-color-4)"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    An LSTM cell uses a <Highlight>gated</Highlight> design. Gates allow information
    to flow or stop information flow depending on the state of the gate.
  </p>
  <p>
    When the gate is closed, no information is allowed to flow and data can not
    pass past the gate.
  </p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 100">
      <Arrow
        data={[
          { x: 0, y: 50 },
          { x: 230, y: 50 },
        ]}
        strokeWidth="4"
        dashed="true"
        strokeDashArray="8 8"
        moving="true"
        speed={20}
      />
      <Block
        x="250"
        y="25"
        width="10"
        height="50"
        color="var(--main-color-1)"
      />
      <Block
        x="250"
        y="75"
        width="10"
        height="50"
        color="var(--main-color-1)"
      />
    </svg>
  </SvgContainer>
  <p>When the gate is open, information can flow without interruption.</p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 100">
      <Arrow
        data={[
          { x: 0, y: 50 },
          { x: 480, y: 50 },
        ]}
        strokeWidth="4"
        dashed="true"
        strokeDashArray="8 8"
        moving="true"
        speed={20}
      />
      <Block
        x="250"
        y="15"
        width="10"
        height="30"
        color="var(--main-color-1)"
      />
      <Block
        x="250"
        y="85"
        width="10"
        height="30"
        color="var(--main-color-1)"
      />
    </svg>
  </SvgContainer>
  <p>
    Essentially an LSTM cell determines which parts of the sequence data can
    flow and should be saved for future reference and which parts are
    irrelevant.
  </p>
  <p>
    While conceptually you probatly understand that gates can stop and allow the
    flow of data, what exactly does that mean mathematically?
  </p>
  <p>
    Let's assume that our data is a two dimensional vecor, with values 2 and 5.
  </p>
  <Latex>
    {String.raw`
        \text{Data} = 
        \begin{bmatrix}
          2 \\
          5 \\ 
        \end{bmatrix}
      `}
  </Latex>
  <p>
    The gate is a vector of the same same, that contains values of either 0 or
    1.
  </p>
  <Latex>
    {String.raw`
        \text{Gate} = 
        \begin{bmatrix}
          1 \\
          0 \\ 
        \end{bmatrix}
      `}
  </Latex>
  <p>
    In order to determine what part of the data is allowed to flow we use
    elementwise multiplication.
  </p>
  <Latex>
    {String.raw`
        \begin{bmatrix}
          2 \\
          5 \\ 
        \end{bmatrix} \odot
        \begin{bmatrix}
          1 \\
          0 \\ 
        \end{bmatrix} = 
        \begin{bmatrix}
          2 \\
          0 \\ 
        \end{bmatrix}
      `}
  </Latex>
  <p>
    The parts of the vector that are multiplied by a 0 are essentially erased,
    while those parts that are multiplied by a 1 are allowed to keep flowing.
  </p>
  <p>
    In practice LSTM cells do not work in a completely binary fashion. Gates can
    contain continuous values between 0 and 1. In the example below you can
    control the "openness" of the gates with a slider. The closer the gate gets
    to a state of 0, the less information is passed through.
  </p>
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 100">
      <Arrow
        data={[
          { x: 0, y: 50 },
          { x: 240, y: 50 },
        ]}
        showMarker={false}
        strokeWidth="70"
        dashed="true"
        strokeDashArray="8 8"
        moving="true"
        speed={20}
      />
      <Arrow
        data={[
          { x: 260, y: 50 },
          { x: 480, y: 50 },
        ]}
        showMarker={false}
        strokeWidth={70 * dummyGateState}
        dashed="true"
        strokeDashArray="8 8"
        moving="true"
        speed={20}
      />
      <Block
        x="250"
        y={-dummyGateState * 40 + 25}
        width="10"
        height="50"
        color="var(--main-color-1)"
      />
      <Block
        x="250"
        y={dummyGateState * 40 + 75}
        width="10"
        height="50"
        color="var(--main-color-1)"
      />
    </svg>
  </SvgContainer>
  <Slider bind:value={dummyGateState} min="0" max="1" step="0.1" />
  <p>
    The beauty of an LSTM cell is its ability to learn and to calculate the
    values of different gates, based on the hidden state and the value of the
    sequence. That means that an LSTM cell decides on the fly which information
    is important for the future and should be saved and which should be
    discarded. These gates are essentially fully connected neural networks,
    which use a sigmid activation function in order to scale the gated values
    between 0 and 1.
  </p>
  <p>
    Now let's have a look at the inner workings of the LSTM cell. The design
    might look intimidating at first glance, but we will take it one step at a
    time.
  </p>
  <SvgContainer>
    <svg viewBox="0 0 500 300">
      <!-- surrounding border -->
      <Border x={50} y={15} width={400} height={230} />

      <!-- long term memory -->
      <Block fontSize={10} x={20} y={40} width={35} height={20} text="C(t-1)" />
      <Block fontSize={10} x={480} y={40} width={35} height={20} text="C(t)" />
      <Arrow
        data={[
          { x: 40, y: 40 },
          { x: 455, y: 40 },
        ]}
        strokeWidth={2}
      />

      <!-- forget gate: multiplication for cell state with forget gate -->
      <Block x={100} y={40} width={15} height={15} color="white" />
      <Multiply x={100} y={40} radius={7} />

      <!-- addition for cell state with input gate -->
      <Block x={175} y={40} width={15} height={15} color="white" />
      <Plus x={175} y={40} radius={7} />

      <!-- short term memory -->
      <Block
        fontSize={10}
        x={20}
        y={230}
        width={35}
        height={20}
        text="H(t-1)"
      />
      <Block fontSize={10} x={480} y={230} width={35} height={20} text="H(t)" />
      <Arrow
        data={[
          { x: 40, y: 230 },
          { x: 300, y: 230 },
          { x: 300, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input -->
      <Block fontSize={10} x={60} y={278} width={35} height={20} text="X(t)" />
      <Arrow
        data={[
          { x: 60, y: 265 },
          { x: 60, y: 230 },
        ]}
        showMarker={false}
        strokeWidth={2}
      />
      <Circle x="60" y="230" r="3" />

      <!-- nn for forget gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={100}
        y={120}
        width={20}
        height={20}
        text="f"
      />
      <Arrow
        data={[
          { x: 100, y: 230 },
          { x: 100, y: 140 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 100, y: 105 },
          { x: 100, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- tanh nn -->
      <Block
        color="var(--main-color-3)"
        fontSize={10}
        x={150}
        y={160}
        width={20}
        height={20}
        text="g"
      />
      <Arrow
        data={[
          { x: 150, y: 230 },
          { x: 150, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for input gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={200}
        y={160}
        width={20}
        height={20}
        text="i"
      />
      <Arrow
        data={[
          { x: 200, y: 230 },
          { x: 200, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input gate: multiplication for i with g -->
      <Block x={175} y={100} width={15} height={15} color="white" />
      <Multiply x={175} y={100} radius={7} />

      <!-- connection from g, i all the way to the addition with c -->
      <Arrow
        data={[
          { x: 150, y: 145 },
          { x: 150, y: 100 },
          { x: 160, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 200, y: 145 },
          { x: 200, y: 100 },
          { x: 190, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 175, y: 90 },
          { x: 175, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for output gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={300}
        y={160}
        width={20}
        height={20}
        text="o"
      />

      <!-- copying of cell state -->
      <Circle x="300" y="40" r="3" />
      <Arrow
        data={[
          { x: 300, y: 40 },
          { x: 300, y: 85 },
        ]}
        strokeWidth={2}
      />

      <!-- output gate -->
      <Block x={300} y={100} width={15} height={15} color="white" />
      <Multiply x={300} y={100} radius={7} />

      <!-- arrow from nn for output gate to output gate -->
      <Arrow
        data={[
          { x: 300, y: 145 },
          { x: 300, y: 115 },
        ]}
        strokeWidth={2}
      />

      <!-- arrow from output gate to the hidden state -->
      <Arrow
        data={[
          { x: 310, y: 100 },
          { x: 350, y: 100 },
          { x: 350, y: 230 },
          { x: 455, y: 230 },
        ]}
        strokeWidth={2}
      />
    </svg>
  </SvgContainer>

  <p>
    The cell outputs the long-term memory <Latex>c_t</Latex> and the short term memory
    <Latex>h_t</Latex>. For that purpose the LSTM cell contains four fully
    connected neural networks. The red networks <Latex>f</Latex>, <Latex
      >i</Latex
    > and <Latex>o</Latex> are networks with a sigmoid activation function, that
    are used to act as gates, while the yellow <Latex>g</Latex> neural network uses
    a tanh activation function and is used to generate "potential memories". All
    four networks take the same inputs: a vector that contains previous hidden state
    <Latex>{String.raw`h_{t-1}`}</Latex> and the current piece of the input <Latex
      >{String.raw`x_{t}`}</Latex
    >.
  </p>
  <p>
    The long term memory is not directly processed by any of the neural
    networks, but can be adjusted through a gate or through additional
    "potential long-term memories". The general idea is to only interrupt the
    flow of the long-term memory, if it is warranted. That allows to establish
    long-term dependencies without relying on the backpropagation algorithm
    directly.
  </p>
  <SvgContainer>
    <svg viewBox="0 0 500 300">
      <!-- Highlight -->
      <Block
        x={250}
        y={40}
        width={500}
        height={30}
        color="rgba(10, 100, 100, 0.2)"
      />
      <!-- surrounding border -->
      <Border x={50} y={15} width={400} height={230} />

      <!-- long term memory -->
      <Block fontSize={10} x={20} y={40} width={35} height={20} text="C(t-1)" />
      <Block fontSize={10} x={480} y={40} width={35} height={20} text="C(t)" />
      <Arrow
        data={[
          { x: 40, y: 40 },
          { x: 455, y: 40 },
        ]}
        strokeWidth={2}
      />

      <!-- forget gate: multiplication for cell state with forget gate -->
      <Block x={100} y={40} width={15} height={15} color="white" />
      <Multiply x={100} y={40} radius={7} />

      <!-- addition for cell state with input gate -->
      <Block x={175} y={40} width={15} height={15} color="white" />
      <Plus x={175} y={40} radius={7} />

      <!-- short term memory -->
      <Block
        fontSize={10}
        x={20}
        y={230}
        width={35}
        height={20}
        text="H(t-1)"
      />
      <Block fontSize={10} x={480} y={230} width={35} height={20} text="H(t)" />
      <Arrow
        data={[
          { x: 40, y: 230 },
          { x: 300, y: 230 },
          { x: 300, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input -->
      <Block fontSize={10} x={60} y={278} width={35} height={20} text="X(t)" />
      <Arrow
        data={[
          { x: 60, y: 265 },
          { x: 60, y: 230 },
        ]}
        showMarker={false}
        strokeWidth={2}
      />
      <Circle x="60" y="230" r="3" />

      <!-- nn for forget gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={100}
        y={120}
        width={20}
        height={20}
        text="f"
      />
      <Arrow
        data={[
          { x: 100, y: 230 },
          { x: 100, y: 140 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 100, y: 105 },
          { x: 100, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- tanh nn -->
      <Block
        color="var(--main-color-3)"
        fontSize={10}
        x={150}
        y={160}
        width={20}
        height={20}
        text="g"
      />
      <Arrow
        data={[
          { x: 150, y: 230 },
          { x: 150, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for input gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={200}
        y={160}
        width={20}
        height={20}
        text="i"
      />
      <Arrow
        data={[
          { x: 200, y: 230 },
          { x: 200, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input gate: multiplication for i with g -->
      <Block x={175} y={100} width={15} height={15} color="white" />
      <Multiply x={175} y={100} radius={7} />

      <!-- connection from g, i all the way to the addition with c -->
      <Arrow
        data={[
          { x: 150, y: 145 },
          { x: 150, y: 100 },
          { x: 160, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 200, y: 145 },
          { x: 200, y: 100 },
          { x: 190, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 175, y: 90 },
          { x: 175, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for output gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={300}
        y={160}
        width={20}
        height={20}
        text="o"
      />

      <!-- copying of cell state -->
      <Circle x="300" y="40" r="3" />
      <Arrow
        data={[
          { x: 300, y: 40 },
          { x: 300, y: 85 },
        ]}
        strokeWidth={2}
      />

      <!-- output gate -->
      <Block x={300} y={100} width={15} height={15} color="white" />
      <Multiply x={300} y={100} radius={7} />

      <!-- arrow from nn for output gate to output gate -->
      <Arrow
        data={[
          { x: 300, y: 145 },
          { x: 300, y: 115 },
        ]}
        strokeWidth={2}
      />

      <!-- arrow from output gate to the hidden state -->
      <Arrow
        data={[
          { x: 310, y: 100 },
          { x: 350, y: 100 },
          { x: 350, y: 230 },
          { x: 455, y: 230 },
        ]}
        strokeWidth={2}
      />
    </svg>
  </SvgContainer>
  <p>
    The gate <Latex>f</Latex> that is used for the long term memory is called the
    <Highlight>forget gate</Highlight>. We multipy each component of the long
    term memory <Latex>{String.raw`c_{t-1}`}</Latex> by each component from the neural
    network <Latex>f</Latex>. As the sigmoid activation produces values between
    0 and 1, this neural network acts as a gate. We can theoretically reduce or
    even completely erase the long term memory if the LSTM cell deems this
    necessary. The closer the outputs of the fully connected neural network are
    to 1, the more long-term memory is kept.
  </p>
  <SvgContainer>
    <svg viewBox="0 0 500 300">
      <!-- Highlight -->
      <Block
        x={100}
        y={130}
        width={20}
        height={200}
        color="rgba(10, 100, 100, 0.2)"
      />
      <!-- surrounding border -->
      <Border x={50} y={15} width={400} height={230} />

      <!-- long term memory -->
      <Block fontSize={10} x={20} y={40} width={35} height={20} text="C(t-1)" />
      <Block fontSize={10} x={480} y={40} width={35} height={20} text="C(t)" />
      <Arrow
        data={[
          { x: 40, y: 40 },
          { x: 455, y: 40 },
        ]}
        strokeWidth={2}
      />

      <!-- forget gate: multiplication for cell state with forget gate -->
      <Block x={100} y={40} width={15} height={15} color="white" />
      <Multiply x={100} y={40} radius={7} />

      <!-- addition for cell state with input gate -->
      <Block x={175} y={40} width={15} height={15} color="white" />
      <Plus x={175} y={40} radius={7} />

      <!-- short term memory -->
      <Block
        fontSize={10}
        x={20}
        y={230}
        width={35}
        height={20}
        text="H(t-1)"
      />
      <Block fontSize={10} x={480} y={230} width={35} height={20} text="H(t)" />
      <Arrow
        data={[
          { x: 40, y: 230 },
          { x: 300, y: 230 },
          { x: 300, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input -->
      <Block fontSize={10} x={60} y={278} width={35} height={20} text="X(t)" />
      <Arrow
        data={[
          { x: 60, y: 265 },
          { x: 60, y: 230 },
        ]}
        showMarker={false}
        strokeWidth={2}
      />
      <Circle x="60" y="230" r="3" />

      <!-- nn for forget gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={100}
        y={120}
        width={20}
        height={20}
        text="f"
      />
      <Arrow
        data={[
          { x: 100, y: 230 },
          { x: 100, y: 140 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 100, y: 105 },
          { x: 100, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- tanh nn -->
      <Block
        color="var(--main-color-3)"
        fontSize={10}
        x={150}
        y={160}
        width={20}
        height={20}
        text="g"
      />
      <Arrow
        data={[
          { x: 150, y: 230 },
          { x: 150, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for input gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={200}
        y={160}
        width={20}
        height={20}
        text="i"
      />
      <Arrow
        data={[
          { x: 200, y: 230 },
          { x: 200, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input gate: multiplication for i with g -->
      <Block x={175} y={100} width={15} height={15} color="white" />
      <Multiply x={175} y={100} radius={7} />

      <!-- connection from g, i all the way to the addition with c -->
      <Arrow
        data={[
          { x: 150, y: 145 },
          { x: 150, y: 100 },
          { x: 160, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 200, y: 145 },
          { x: 200, y: 100 },
          { x: 190, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 175, y: 90 },
          { x: 175, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for output gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={300}
        y={160}
        width={20}
        height={20}
        text="o"
      />

      <!-- copying of cell state -->
      <Circle x="300" y="40" r="3" />
      <Arrow
        data={[
          { x: 300, y: 40 },
          { x: 300, y: 85 },
        ]}
        strokeWidth={2}
      />

      <!-- output gate -->
      <Block x={300} y={100} width={15} height={15} color="white" />
      <Multiply x={300} y={100} radius={7} />

      <!-- arrow from nn for output gate to output gate -->
      <Arrow
        data={[
          { x: 300, y: 145 },
          { x: 300, y: 115 },
        ]}
        strokeWidth={2}
      />

      <!-- arrow from output gate to the hidden state -->
      <Arrow
        data={[
          { x: 310, y: 100 },
          { x: 350, y: 100 },
          { x: 350, y: 230 },
          { x: 455, y: 230 },
        ]}
        strokeWidth={2}
      />
    </svg>
  </SvgContainer>
  <p>
    In the second step we decide if we should add anything to the long term
    memory. First we calculate the memories that can be used as potential
    additions to the long-term memory using the fully connected neural network <Latex
      >{String.raw`g`}</Latex
    >. Additionally we use the neural network <Latex>{String.raw`i`}</Latex>,
    which acts as a gate for those "potential memories". This gate is called <Highlight
      >input gate</Highlight
    >. The elementwise product of the two neural networks outputs is the actual
    adjustment to the long-term state, which are added elementwise to the values
    that were passed through the forget gate.
  </p>
  <SvgContainer>
    <svg viewBox="0 0 500 300">
      <!-- Highlight -->
      <Block
        x={175}
        y={130}
        width={70}
        height={200}
        color="rgba(10, 100, 100, 0.2)"
      />
      <!-- surrounding border -->
      <Border x={50} y={15} width={400} height={230} />

      <!-- long term memory -->
      <Block fontSize={10} x={20} y={40} width={35} height={20} text="C(t-1)" />
      <Block fontSize={10} x={480} y={40} width={35} height={20} text="C(t)" />
      <Arrow
        data={[
          { x: 40, y: 40 },
          { x: 455, y: 40 },
        ]}
        strokeWidth={2}
      />

      <!-- forget gate: multiplication for cell state with forget gate -->
      <Block x={100} y={40} width={15} height={15} color="white" />
      <Multiply x={100} y={40} radius={7} />

      <!-- addition for cell state with input gate -->
      <Block x={175} y={40} width={15} height={15} color="white" />
      <Plus x={175} y={40} radius={7} />

      <!-- short term memory -->
      <Block
        fontSize={10}
        x={20}
        y={230}
        width={35}
        height={20}
        text="H(t-1)"
      />
      <Block fontSize={10} x={480} y={230} width={35} height={20} text="H(t)" />
      <Arrow
        data={[
          { x: 40, y: 230 },
          { x: 300, y: 230 },
          { x: 300, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input -->
      <Block fontSize={10} x={60} y={278} width={35} height={20} text="X(t)" />
      <Arrow
        data={[
          { x: 60, y: 265 },
          { x: 60, y: 230 },
        ]}
        showMarker={false}
        strokeWidth={2}
      />
      <Circle x="60" y="230" r="3" />

      <!-- nn for forget gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={100}
        y={120}
        width={20}
        height={20}
        text="f"
      />
      <Arrow
        data={[
          { x: 100, y: 230 },
          { x: 100, y: 140 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 100, y: 105 },
          { x: 100, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- tanh nn -->
      <Block
        color="var(--main-color-3)"
        fontSize={10}
        x={150}
        y={160}
        width={20}
        height={20}
        text="g"
      />
      <Arrow
        data={[
          { x: 150, y: 230 },
          { x: 150, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for input gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={200}
        y={160}
        width={20}
        height={20}
        text="i"
      />
      <Arrow
        data={[
          { x: 200, y: 230 },
          { x: 200, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input gate: multiplication for i with g -->
      <Block x={175} y={100} width={15} height={15} color="white" />
      <Multiply x={175} y={100} radius={7} />

      <!-- connection from g, i all the way to the addition with c -->
      <Arrow
        data={[
          { x: 150, y: 145 },
          { x: 150, y: 100 },
          { x: 160, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 200, y: 145 },
          { x: 200, y: 100 },
          { x: 190, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 175, y: 90 },
          { x: 175, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for output gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={300}
        y={160}
        width={20}
        height={20}
        text="o"
      />

      <!-- copying of cell state -->
      <Circle x="300" y="40" r="3" />
      <Arrow
        data={[
          { x: 300, y: 40 },
          { x: 300, y: 85 },
        ]}
        strokeWidth={2}
      />

      <!-- output gate -->
      <Block x={300} y={100} width={15} height={15} color="white" />
      <Multiply x={300} y={100} radius={7} />

      <!-- arrow from nn for output gate to output gate -->
      <Arrow
        data={[
          { x: 300, y: 145 },
          { x: 300, y: 115 },
        ]}
        strokeWidth={2}
      />

      <!-- arrow from output gate to the hidden state -->
      <Arrow
        data={[
          { x: 310, y: 100 },
          { x: 350, y: 100 },
          { x: 350, y: 230 },
          { x: 455, y: 230 },
        ]}
        strokeWidth={2}
      />
    </svg>
  </SvgContainer>

  <p>
    The final neural network <Latex>o</Latex> is used to determine which values are
    suitable for the short-term memory <Latex>h_t</Latex>. This gate is called
    the <Highlight>output gate</Highlight>. For that purpose the long-term
    memory <Latex>c_t</Latex> is copied and is processed by the tanh activation function.
    The result is passed through the output gate by the means of elementwise multiplication.
  </p>
  <SvgContainer>
    <svg viewBox="0 0 500 300">
      <!-- Highlight -->
      <Block
        x={300}
        y={130}
        width={20}
        height={200}
        color="rgba(10, 100, 100, 0.2)"
      />
      <!-- surrounding border -->
      <Border x={50} y={15} width={400} height={230} />

      <!-- long term memory -->
      <Block fontSize={10} x={20} y={40} width={35} height={20} text="C(t-1)" />
      <Block fontSize={10} x={480} y={40} width={35} height={20} text="C(t)" />
      <Arrow
        data={[
          { x: 40, y: 40 },
          { x: 455, y: 40 },
        ]}
        strokeWidth={2}
      />

      <!-- forget gate: multiplication for cell state with forget gate -->
      <Block x={100} y={40} width={15} height={15} color="white" />
      <Multiply x={100} y={40} radius={7} />

      <!-- addition for cell state with input gate -->
      <Block x={175} y={40} width={15} height={15} color="white" />
      <Plus x={175} y={40} radius={7} />

      <!-- short term memory -->
      <Block
        fontSize={10}
        x={20}
        y={230}
        width={35}
        height={20}
        text="H(t-1)"
      />
      <Block fontSize={10} x={480} y={230} width={35} height={20} text="H(t)" />
      <Arrow
        data={[
          { x: 40, y: 230 },
          { x: 300, y: 230 },
          { x: 300, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input -->
      <Block fontSize={10} x={60} y={278} width={35} height={20} text="X(t)" />
      <Arrow
        data={[
          { x: 60, y: 265 },
          { x: 60, y: 230 },
        ]}
        showMarker={false}
        strokeWidth={2}
      />
      <Circle x="60" y="230" r="3" />

      <!-- nn for forget gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={100}
        y={120}
        width={20}
        height={20}
        text="f"
      />
      <Arrow
        data={[
          { x: 100, y: 230 },
          { x: 100, y: 140 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 100, y: 105 },
          { x: 100, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- tanh nn -->
      <Block
        color="var(--main-color-3)"
        fontSize={10}
        x={150}
        y={160}
        width={20}
        height={20}
        text="g"
      />
      <Arrow
        data={[
          { x: 150, y: 230 },
          { x: 150, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for input gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={200}
        y={160}
        width={20}
        height={20}
        text="i"
      />
      <Arrow
        data={[
          { x: 200, y: 230 },
          { x: 200, y: 180 },
        ]}
        strokeWidth={2}
      />

      <!-- input gate: multiplication for i with g -->
      <Block x={175} y={100} width={15} height={15} color="white" />
      <Multiply x={175} y={100} radius={7} />

      <!-- connection from g, i all the way to the addition with c -->
      <Arrow
        data={[
          { x: 150, y: 145 },
          { x: 150, y: 100 },
          { x: 160, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 200, y: 145 },
          { x: 200, y: 100 },
          { x: 190, y: 100 },
        ]}
        strokeWidth={2}
      />
      <Arrow
        data={[
          { x: 175, y: 90 },
          { x: 175, y: 55 },
        ]}
        strokeWidth={2}
      />

      <!-- nn for output gate -->
      <Block
        color="var(--main-color-1)"
        fontSize={10}
        x={300}
        y={160}
        width={20}
        height={20}
        text="o"
      />

      <!-- copying of cell state -->
      <Circle x="300" y="40" r="3" />
      <Arrow
        data={[
          { x: 300, y: 40 },
          { x: 300, y: 85 },
        ]}
        strokeWidth={2}
      />

      <!-- output gate -->
      <Block x={300} y={100} width={15} height={15} color="white" />
      <Multiply x={300} y={100} radius={7} />

      <!-- arrow from nn for output gate to output gate -->
      <Arrow
        data={[
          { x: 300, y: 145 },
          { x: 300, y: 115 },
        ]}
        strokeWidth={2}
      />

      <!-- arrow from output gate to the hidden state -->
      <Arrow
        data={[
          { x: 310, y: 100 },
          { x: 350, y: 100 },
          { x: 350, y: 230 },
          { x: 455, y: 230 },
        ]}
        strokeWidth={2}
      />
    </svg>
  </SvgContainer>
</Container>
<Footer {references} />
