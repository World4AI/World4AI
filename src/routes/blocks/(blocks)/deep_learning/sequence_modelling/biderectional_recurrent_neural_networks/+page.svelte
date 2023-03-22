<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
</script>

<svelte:head>
  <title>Bidirectional Recurrent Neural Network - World4AI</title>
  <meta
    name="description"
    content="Unlike a plain vanilla recurrent neural network, a biderectional rnn traverses the sequence in two directions. From front to back and from back to front. The output concatenates the two sets of hidden units."
  />
</svelte:head>

<h1>Biderectional Recurrent Neural Networks</h1>
<div class="separator" />

<Container>
  <p>
    A recurrent neural network processes one part of the sequence at a time.
    When we are dealing with a sentence for example, the neural network starts
    with the very first word and moves forward through the sentence. A
    <Highlight>biderectional recurrent neural network</Highlight> traverses the sequence
    from two directions. As usual from the start to finish and in the reverse direction,
    from finish to start. The output of the network, <Latex
      >{String.raw`\mathbf{y_t}`}</Latex
    >, simply concatenates the two vectors that come from different directions.
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
              { x: 112, y: 145 },
              { x: 112, y: 68 },
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
            text="h_{idx + 1}"
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
    A biderectional recurrent neural network is especially well suited for
    language tasks. Look at the two sentences below.
  </p>
  <p class="bg-slate-200 py-2 px-0 text-center rounded-xl">
    The bank opens ...
  </p>
  <p class="bg-red-200 py-2 px-0 text-center rounded-xl">
    The bank of the river ...
  </p>
  <p>
    While the sentences start out with the same two words, the meaning can only
    be understood by reading through the whole sentence.
  </p>
  <p>
    A biderectional RNN is not suited for every task though. If you intend to
    predict future points of a time series data and you use a biderectional RNN,
    you will introduce <Highlight>data leakage</Highlight>. Data leakage means
    that during training your network has access to the type of information,
    that is not available during inference. Using a biderectional RNN would
    imply that you use future time series information to train your neural
    network, like training a RNN to predict the stock price, that the network
    has already observed.
  </p>
  <p>
    We can implement a biderectional RNN in PyTorch by simply setting the <code
      >bidirectional</code
    >
    flag to <code>True</code>.
  </p>
  <PythonCode
    code={`batch_size=4
sequence_length=5
input_size=6
hidden_size=3
num_layers=2`}
  />
  <PythonCode
    code={`rnn = nn.RNN(input_size=input_size, 
             hidden_size=hidden_size, 
             num_layers=num_layers, 
             bidirectional=True)`}
  />
  <PythonCode
    code={`sequence = torch.randn(sequence_length, batch_size, input_size)
# 2*num_layers due to biderectional model
h_0 = torch.zeros(2*num_layers, batch_size, hidden_size)`}
  />
  <p>
    Due to the biderectional nature of the recurrent neural network, the
    dimensions of the outputs and the hidden states increase.
  </p>
  <PythonCode
    code={`with torch.inference_mode():
    output, h_n = rnn(sequence, h_0)
print(output.shape, h_n.shape)`}
  />
  <PythonCode
    code={`torch.Size([5, 4, 6]) torch.Size([4, 4, 3])`}
    isOutput={true}
  />

  <div class="separator" />
</Container>
