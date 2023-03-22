<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
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
</script>

<svelte:head>
  <title>LSTM - World4AI</title>
  <meta
    name="description"
    content="The LSTM (long short-term memory) cell uses a gated architecture in order to learn and preserve long-term dependencies. Unlike the simple RNN neural network, LSTM networks can deal with large sequences."
  />
</svelte:head>

<h1>LSTM</h1>
<div class="separator" />

<Container>
  <p>
    Often a recurrent neural network has to process very large sequences. With
    each recurrent step we incorporate more and more information into the hidden
    state while the distance to the start of the sequence increases. Let's take
    the following language task as an example.
  </p>
  <p class="bg-blue-200 px-3 py-2">
    <Highlight>Computers</Highlight> have been a passion of mine from the very early
    age. I was fascinated by the idea that I could create anything out of nothing
    using only my skills and my imagination. So when the time came to select my major,
    I did not hesitate and picked <Highlight>-----------</Highlight>.
  </p>
  <p>
    In this task we need to predict the last word(s). The key to the solution is
    to realize that the very first word in the very first sentence provides the
    necessary context. If that person has been interested in computers for such
    a long time, it is reasonable to assume that <em>computer science</em>
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
    less attractive. A recurrent neural network shares weights, thus when the
    network encounters a long sequence, gradients will explode or vanish. We can
    use gradient clipping to deal with exploding gradients, but many of the
    techniques that we used with feed-forward neural networks to deal with
    vanishing gradients will not work. Batch normalization for example
    calculates the mean and the standard deviation per feature and layer, but in
    a recurrent neural network the weights are shared and we might theoretically
    need different statistics for a different part of the recurrent loop.
    Specialized techniques were developed to deal with the vanishing gradients
    problem of RNNs and in this section we are gong to cover a new type of a
    recurrent neural network called long short-term memory, or <Highlight
      >LSTM</Highlight
    ><InternalLink type="reference" id={1} />
    for short.
  </p>
  <p>
    For the most part we do not change the overarching design of a recurrent
    neural network. The LSTM cell produces outputs, that are used as an input in
    the next iteration. Unlike the regular RNN cell, an LSTM cell produces two
    vectors: the short term memory <Latex>{String.raw`\mathbf{h}_t`}</Latex> (hidden
    value) and the long term memory
    <Latex>{String.raw`\mathbf{c_t}`}</Latex>(cell value).
  </p>
  <SvgContainer maxWidth={"250px"}>
    <svg viewBox="0 0 200 470">
      {#each Array(4) as _, idx}
        <g transform="translate(0, {idx * 120 - 20})">
          <Arrow
            strokeWidth="2"
            data={[
              { x: 31, y: 45 },
              { x: 76, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 120, y: 45 },
              { x: 164, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 87, y: 62 },
              { x: 87, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 112, y: 62 },
              { x: 112, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Block x="100" y="45" width="30" height="30" class="fill-slate-500" />
          <Block
            text="x_{idx + 1}"
            type="latex"
            fontSize={12}
            x="15"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="y_{idx + 1}"
            fontSize={12}
            x="185"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="80"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
          <Block
            type="latex"
            text="c_{idx + 1}"
            fontSize={12}
            x="115"
            y="100"
            width="25"
            height="25"
            class="fill-red-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    An LSTM cell makes heavy use of so-called <Highlight>gates</Highlight>.
    Gates allow information to keep flowing or stop information flow depending
    on the state of the gate.
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
      <Block x="250" y="25" width="10" height="50" class="fill-slate-400" />
      <Block x="250" y="75" width="10" height="50" class="fill-slate-400" />
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
      <Block x="250" y="15" width="10" height="30" class="fill-slate-400" />
      <Block x="250" y="85" width="10" height="30" class="fill-slate-400" />
    </svg>
  </SvgContainer>
  <p>
    Essentially an LSTM cell determines which parts of the sequence data should
    be processed and saved for future reference and which parts are irrelevant.
  </p>
  <p>
    Let's for example assume that our data is a two dimensional vector, with
    values 2 and 5.
  </p>
  <div class="flex justify-center">
    <Latex>
      {String.raw`
        \text{Data} = 
        \begin{bmatrix}
          2 \\
          5 \\ 
        \end{bmatrix}
      `}
    </Latex>
  </div>
  <p>
    The gate is a vector of the same size, that contains values of either 0 or
    1.
  </p>
  <div class="flex justify-center">
    <Latex>
      {String.raw`
        \text{Gate} = 
        \begin{bmatrix}
          1 \\
          0 \\ 
        \end{bmatrix}
      `}
    </Latex>
  </div>
  <p>
    In order to determine what part of the data is allowed to flow we use
    elementwise multiplication.
  </p>
  <div class="flex justify-center">
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
  </div>
  <p>
    The parts of the vector that are multiplied by a 0 are essentially erased,
    while those parts that are multiplied by a 1 are allowed to keep flowing. In
    practice LSTM cells do not work in a completely binary fashion, but contain
    continuous values between 0 and 1. This allows the gate to pass just a
    fraction of the information.
  </p>
  <p>
    The beauty of an LSTM cell is its ability to learn and to calculate the
    values of different gates automatically. That means that an LSTM cell
    decides on the fly which information is important for the future and should
    be saved and which information should be discarded. These gates are
    essentially fully connected neural networks, which use a sigmid activation
    function in order to scale the values between 0 and 1.
  </p>
  <p>
    Now let's have a look at the inner workings of the LSTM cell. The design
    might look intimidating at first glance, but we will take it one step at a
    time.
  </p>
  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 300 450">
      <!-- C_t-1 -->
      <Block
        fontSize={12}
        x={280}
        y={20}
        width={30}
        height={30}
        text={`\\mathbf{c}_{t-1}`}
        type="latex"
        class="fill-green-100"
      />

      <!-- H_t-1 -->
      <Block
        fontSize={12}
        x={80}
        y={20}
        width={30}
        height={30}
        text={`\\mathbf{h}_{t-1}`}
        type="latex"
        class="fill-green-100"
      />

      <!-- X_t-1 -->
      <Block
        fontSize={12}
        x={20}
        y={60}
        width={30}
        height={30}
        text={`\\mathbf{x}_{t}`}
        type="latex"
        class="fill-green-100"
      />

      <!-- merge of x with h line -->
      <Arrow
        data={[
          { x: 40, y: 60 },
          { x: 80, y: 60 },
        ]}
        showMarker={false}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- long term memory arrow -->
      <Arrow
        data={[
          { x: 280, y: 40 },
          { x: 280, y: 430 },
        ]}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- forget gate: multiplication for cell state with forget gate -->
      <Multiply x={280} y={90} radius={10} class="fill-blue-200" />

      <!-- short term memory arrow -->
      <Arrow
        data={[
          { x: 80, y: 40 },
          { x: 80, y: 430 },
        ]}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- Merging of x and h -->
      <Circle x={80} y={60} r={5} class="fill-black" />

      <!-- forget arrow -->
      <Arrow
        data={[
          { x: 80, y: 90 },
          { x: 260, y: 90 },
        ]}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- nn for forget gate -->
      <Block
        fontSize={16}
        x={180}
        y={90}
        width={30}
        height={30}
        text="f"
        type="latex"
        class="fill-red-400"
      />

      <!-- addition for cell state with input gate -->
      <Plus x={280} y={180} radius={10} offset={4} class="fill-blue-200" />

      <!-- tanh nn arrow -->
      <Arrow
        data={[
          { x: 80, y: 160 },
          { x: 200, y: 160 },
          { x: 200, y: 180 },
        ]}
        showMarker={false}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- tanh nn -->
      <Block
        fontSize={16}
        x={140}
        y={160}
        width={30}
        height={30}
        text="g"
        type="latex"
        class="fill-violet-200"
      />

      <!-- arrow for input gate -->
      <Arrow
        data={[
          { x: 80, y: 210 },
          { x: 200, y: 210 },
          { x: 200, y: 180 },
        ]}
        showMarker={false}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />
      <!-- nn for input gate -->
      <Block
        fontSize={16}
        x={140}
        y={210}
        width={30}
        height={30}
        text="i"
        type="latex"
        class="fill-red-400"
      />

      <!-- input gate: multiplication for i with g -->
      <Multiply x={200} y={180} radius={10} class="fill-blue-200" />

      <!-- arrow from input gate to addition -->
      <Arrow
        data={[
          { x: 210, y: 180 },
          { x: 260, y: 180 },
        ]}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- nn for output gate -->
      <Block
        fontSize={16}
        x={80}
        y={280}
        width={30}
        height={30}
        text="o"
        type="latex"
        class="fill-red-400"
      />

      <!-- output gate: multiplication for tanh cell with output gate -->
      <Multiply x={80} y={350} radius={10} class="fill-blue-200" />

      <!-- arrow from cell state to addition -->
      <Arrow
        data={[
          { x: 210, y: 180 },
          { x: 260, y: 180 },
        ]}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!--  arrow from long term to output gate -->
      <Arrow
        data={[
          { x: 280, y: 350 },
          { x: 100, y: 350 },
        ]}
        dashed={true}
        strokeDashArray="5 5"
        moving={true}
        strokeWidth={2}
      />

      <!-- simple tanh -->
      <Block
        fontSize={13}
        x={180}
        y={350}
        width={50}
        height={30}
        text="\\tanh"
        type="latex"
        class="fill-gray-100"
      />

      <!-- C_t -->
      <Block
        fontSize={12}
        x={280}
        y={400}
        width={30}
        height={30}
        text={`\\mathbf{c}_{t}`}
        type="latex"
        class="fill-green-100"
      />

      <!-- H_t -->
      <Block
        fontSize={12}
        x={80}
        y={400}
        width={30}
        height={30}
        text={`\\mathbf{h}_{t}`}
        type="latex"
        class="fill-green-100"
      />
    </svg>
  </SvgContainer>

  <p>
    The LSTM cell outputs the long-term memory <Latex
      >{String.raw`\mathbf{c}_t`}</Latex
    > and the short term memory
    <Latex>{String.raw`\mathbf{h}_t`}</Latex>. For that purpose the LSTM cell
    contains four fully connected neural networks. The red networks <Latex
      >f</Latex
    >, <Latex>i</Latex> and <Latex>o</Latex> are networks with a sigmoid activation
    function, that act as gates, while the violet neural network <Latex>g</Latex
    > applies a tanh activation function and is used to generate values that can
    be used to adjust the long-term memory. All four networks take the same inputs:
    a vector that contains previous hidden state
    <Latex>{String.raw`\mathbf{h}_{t-1}`}</Latex> and the current piece of the sequence
    <Latex>{String.raw`\mathbf{x}_{t}`}</Latex>.
  </p>
  <p>
    If you look at the flow of the long term memory <Latex
      >{String.raw`\mathbf{c}`}</Latex
    > you should notice, that it flows in a straight line from one part of the sequence
    to the next. The general idea is to only adjust that flow if it is warranted.
    That allows the LSTM cell to establish long-term dependencies.
  </p>
  <p>
    The neural network <Latex>f</Latex> calculates the
    <Highlight>forget gate</Highlight>. We multipy each component of the long
    term memory <Latex>{String.raw`\mathbf{c}_{t-1}`}</Latex> vector by each component
    from the neural network <Latex>f</Latex>. The gate uses the sigmoid
    activation function and can therefore theoretically reduce or even
    completely erase the long term memory if the LSTM cell deems this necessary.
    The closer the outputs of the fully connected neural network are to 1, the
    more long-term memory is kept.
  </p>

  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{f_t} = \sigma(\mathbf{h_{t-1}}\mathbf{W}_{hf}^T + \mathbf{x_t} \mathbf{W}_{xf}^T + \mathbf{b}_f)`}</Latex
    >
  </div>
  <p>
    In the second step we decide if we should add anything to the long term
    memory. First we calculate the memories that can be used as potential
    additions to the long-term memory using the fully connected neural network <Latex
      >{String.raw`g`}</Latex
    >.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{g_t} = \tanh(\mathbf{h_{t-1}}\mathbf{W}_{hg}^T + \mathbf{x_t} \mathbf{W}_{xg}^T + \mathbf{b}_g)`}</Latex
    >
  </div>
  <p>
    Then we use the neural network <Latex>{String.raw`i`}</Latex>, which acts as
    a gate for those "potential memories". This gate is called <Highlight
      >input gate</Highlight
    >. The elementwise product of the two neural networks outputs is the actual
    adjustment to the long-term state, which are added elementwise to the values
    that were passed through the forget gate.
  </p>

  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{i_t} = \sigma(\mathbf{h_{t-1}}\mathbf{W}_{hi}^T + \mathbf{x_t} \mathbf{W}_{xi}^T + \mathbf{b}_i)`}</Latex
    >
  </div>
  <p>
    The forget gate, the input gate and the "potential memories" are used to
    calculate the long-term memories for the next timestep of the series.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{c_t} = \mathbf{f_t} \odot \mathbf{c}_{t-1} + \mathbf{i}_{t} \odot \mathbf{g}_t `}</Latex
    >
  </div>
  <p>
    The final neural network <Latex>o</Latex> is used to determine which values are
    suitable for the short-term memory <Latex>{String.raw`\mathbf{h}_t`}</Latex
    >. This gate is called the <Highlight>output gate</Highlight>. For that
    purpose the long-term memory <Latex>{String.raw`\mathbf{c}_t`}</Latex> is copied
    and is preprocessed by the tanh activation function. The result is multiplied
    by the output gate.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{o_t} = \sigma(\mathbf{h_{t-1}}\mathbf{W}_{ho}^T + \mathbf{x_t} \mathbf{W}_{xo}^T + \mathbf{b}_o)`}</Latex
    >
  </div>
  <p />
  <div class="flex justify-center">
    <Latex
      >{String.raw`\mathbf{h_t} = \mathbf{o_t} \odot \tanh(\mathbf{c}_{t})`}</Latex
    >
  </div>
  <p>
    If we want to use a LSTM instead of a plain valilla recurrent neural net, we
    have to use the <a
      href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"
      target="_blank"
      rel="noreferrer"><code>nn.LSTM</code></a
    >
    module instead of the
    <code>nn.RNN</code>.
  </p>
  <PythonCode
    code={`batch_size=4
sequence_length=5
input_size=6
hidden_size=3
num_layers=2`}
  />
  <PythonCode
    code={`lstm = nn.LSTM(input_size=input_size, 
               hidden_size=hidden_size, 
               num_layers=num_layers)`}
  />
  <p>
    We need to account for the long term memory, but the rest of the
    implementation is almost identical.
  </p>
  <PythonCode
    code={`# create inputs to the LSTM
sequence = torch.randn(sequence_length, batch_size, input_size)
h_0 = torch.zeros(num_layers, batch_size, hidden_size)
c_0 = torch.zeros(num_layers, batch_size, hidden_size)`}
  />
  <PythonCode
    code={`with torch.inference_mode():
    output, (h_n, c_n) = lstm(sequence, (h_0, c_0))
print(output.shape, h_n.shape, c_n.shape)`}
  />
  <PythonCode
    code={`torch.Size([5, 4, 3]) torch.Size([2, 4, 3]) torch.Size([2, 4, 3])`}
    isOutput={true}
  />
</Container>
<Footer {references} />
