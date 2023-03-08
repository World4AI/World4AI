<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Circle from "$lib/diagram/Circle.svelte";

  // button
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  import rnn from "./rnn.png";
  import sine from "./sine.png";

  // coordinates of the circles in the fc diagram
  let fcCirclex = 100;
  let fcCircley = 50;
  function simulateSingleInput() {
    fcCirclex += 1;
    if (fcCirclex >= 400) {
      fcCirclex = 100;
    }
  }

  let fcCircle1x = 10;
  let fcCircle2x = 10;
  let fcCircle1y = 50;
  let fcCircle2y = 150;

  function simulateFc() {
    fcCircle1x += 1;
    fcCircle2x += 1;
    if (fcCircle1x >= 130 && fcCircle1y < 100.5) {
      fcCircle1y += 0.45;
      fcCircle2y -= 0.45;
    }

    if (fcCircle1x > 225) {
      fcCircle1y = 100;
      fcCircle2y = 100;
    }

    if (fcCircle1x >= 350) {
      fcCircle1x = 10;
      fcCircle2x = 10;
      fcCircle1y = 50;
      fcCircle2y = 150;
    }
  }

  // coordinates of the circles in the rnn diagram
  let rnnCircle1x = 10;
  let rnnCircle2x = 10;
  let rnnCircle1y = 50;
  let rnnCircle2y = 150;
  let loops = 3;
  let loop = 0;

  let turnAround = false;
  function simulateRnn() {
    // move forward
    if (!turnAround) {
      // move forward from start to declining arrow
      rnnCircle1x += 1;
      rnnCircle2x += 1;

      //  when you reach a declining arrow, start moving in the y direction as well
      if (rnnCircle1x >= 130 && rnnCircle1y < 100.5) {
        rnnCircle1y += 0.45;
        rnnCircle2y -= 0.45;
      }

      // join the two circles
      if (rnnCircle1x > 225) {
        rnnCircle1y = 100;
        rnnCircle2y = 100;
      }

      if (rnnCircle1x >= 250 && loop < loops) {
        turnAround = true;
      }
      // finished
      if (rnnCircle1x >= 350) {
        rnnCircle1x = 10;
        rnnCircle2x = 10;
        rnnCircle1y = 50;
        rnnCircle2y = 150;
        loop = 0;
      }
    }
    // loop around
    else {
      // move below
      if (rnnCircle1y < 190 && rnnCircle1x >= 250) {
        rnnCircle1y += 1;
        rnnCircle2y += 1;
      } else if (rnnCircle1y >= 190 && rnnCircle1x > 10) {
        // move left
        rnnCircle1x -= 1;
        rnnCircle2x -= 1;
      } else if (rnnCircle1x <= 10 && rnnCircle1y > 150) {
        // move top
        rnnCircle1y -= 1;
        rnnCircle2y -= 1;
      } else {
        turnAround = false;
        rnnCircle1x = 10;
        rnnCircle2x = 10;
        rnnCircle1y = 50;
        rnnCircle2y = 150;
        loop += 1;
      }
    }
  }
</script>

<svelte:head>
  <title>Recurrent Neural Networks - World4AI</title>
  <meta
    name="description"
    content="A recurrent neural network is very well suited to process sequential data. The RNN processes one piece at a time and keeps the outputs from the previous parts of the sequence in the memory which is used as an additional input in the next step."
  />
</svelte:head>

<h1>Recurrent Neural Networks</h1>
<div class="separator" />

<Container>
  <p>
    Let's start this section by contrasting and comparing a plain vanilla feed
    forward neural network with a recurrent neural network.
  </p>
  <p>
    Let's assume for the moment that we are dealing with a single neuron that
    receives a single input. This input could for example be the current
    temperature level and our prediction is the temperature for the next day.
    The feedforward neural network processes the input and generates the output.
    Once the input has left the neuron it is forgotten. This neuron has no
    memory.
  </p>
  <ButtonContainer>
    <PlayButton f={simulateSingleInput} delta={5} />
  </ButtonContainer>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 400 100">
      <!-- connections -->
      <Arrow
        strokeWidth="2"
        data={[
          { x: 130, y: 50 },
          { x: 215, y: 50 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 280, y: 50 },
          { x: 380, y: 50 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <!-- neurons -->
      <Block x="100" y="50" width="20" height="20" class="fill-slate-300" />
      <Block x="250" y="50" width="50" height="50" class="fill-slate-600" />
      <!-- moving data -->
      <Circle x={fcCirclex} y={fcCircley} r="5" class="fill-yellow-400" />
    </svg>
  </SvgContainer>
  <p>
    When the model is dealing with sequences, it should probably remember at
    least some parts of the previous inputs. The meaning of a sentence for
    example depends on the understanding of the whole sentence and not a single
    word. A similar argument can be made for the prediction of the temperature.
    It would probably be useful for the model to remember the temperature of the
    previous couple of days. We could try to circumvent the problem by adding
    additional neuron. Two neurons for example could be used to represent the
    temperature from the past day and the day before that. The output of the two
    neurons would be passed to the next layer.
  </p>
  <!-- fc diagram -->
  <ButtonContainer>
    <PlayButton f={simulateFc} delta={5} />
  </ButtonContainer>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 400 200">
      <!-- connections -->
      <Arrow
        strokeWidth="2"
        data={[
          { x: 5, y: 50 },
          { x: 65, y: 50 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 5, y: 150 },
          { x: 65, y: 150 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 130, y: 50 },
          { x: 215, y: 90 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 130, y: 150 },
          { x: 215, y: 110 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="2"
        data={[
          { x: 280, y: 100 },
          { x: 380, y: 100 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <!-- neurons -->
      <Block x="100" y="50" width="20" height="20" class="fill-slate-300" />
      <Block x="100" y="150" width="20" height="20" class="fill-slate-300" />
      <Block x="250" y="100" width="50" height="50" class="fill-slate-600" />
      <!-- moving data -->
      <Circle x={fcCircle1x} y={fcCircle1y} r="5" class="fill-yellow-400" />
      <Circle x={fcCircle2x} y={fcCircle2y} r="5" class="fill-yellow-400" />
    </svg>
  </SvgContainer>
  <p>
    The above approach does not completely solve the problem though. Many
    sequences have a variable length. The length of a sentence that we would
    like to translate for example can change dramatically. We need a more
    flexible system. A <Highlight>recurrent neural network</Highlight> offers a way
    out.
  </p>
  <p>
    A recurrent neural network (often abbreviated as RNN) processes each piece
    of a sequence at a time. At each time step the neuron takes a part of the
    sequence and its own output from the previous timestep as input. In the very
    first time step there is no output from the previous step, so it is common
    to use 0 instead.
  </p>
  <p>
    Below for example we are dealing with a sequence of size 4. This could for
    example be temperature measuremets from the 4 previous days. Once the
    sequence is exhaused, the output is sent to the next unit, for example the
    next recurrent layer.
  </p>
  <!-- rnn diagram -->
  <ButtonContainer>
    <PlayButton f={simulateRnn} delta={5} />
  </ButtonContainer>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 400 250">
      {#each Array(4) as _, idx}
        <Block
          x={50 + idx * 40}
          y="30"
          width="20"
          height="20"
          text={idx + 1}
          fontSize={15}
          color={loop === idx ? "var(--main-color-1)" : "var(--main-color-4)"}
        />
      {/each}
      <g transform="translate(0 50)">
        <!-- connections -->
        <Arrow
          strokeWidth="2"
          data={[
            { x: 5, y: 50 },
            { x: 65, y: 50 },
          ]}
          dashed={true}
          moving={true}
          strokeDashArray="4 4"
        />
        <Arrow
          strokeWidth="2"
          data={[
            { x: 250, y: 125 },
            { x: 250, y: 190 },
            { x: 10, y: 190 },
            { x: 10, y: 150 },
            { x: 65, y: 150 },
          ]}
          dashed={true}
          moving={true}
          strokeDashArray="4 4"
        />
        <Arrow
          strokeWidth="2"
          data={[
            { x: 130, y: 50 },
            { x: 215, y: 90 },
          ]}
          dashed={true}
          moving={true}
          strokeDashArray="4 4"
        />
        <Arrow
          strokeWidth="2"
          data={[
            { x: 130, y: 150 },
            { x: 215, y: 110 },
          ]}
          dashed={true}
          moving={true}
          strokeDashArray="4 4"
        />
        <Arrow
          strokeWidth="2"
          data={[
            { x: 280, y: 100 },
            { x: 380, y: 100 },
          ]}
          dashed={true}
          moving={true}
          strokeDashArray="4 4"
        />

        <!-- neurons -->
        <Block x="100" y="50" width="20" height="20" class="fill-slate-300" />
        <Block x="100" y="150" width="20" height="20" class="fill-slate-300" />
        <Block x="250" y="100" width="50" height="50" class="fill-slate-500" />

        <!-- moving data -->
        <Circle
          x={rnnCircle1x}
          y={rnnCircle1y}
          r="5"
          color="var(--main-color-1)"
        />
        <Circle
          x={rnnCircle2x}
          y={rnnCircle2y}
          r="5"
          color="var(--main-color-1)"
        />
      </g>
    </svg>
  </SvgContainer>
  <p>
    When you start to study recurrent neural networks, you might encounter a
    specific visual notation for RNNs, similar to the one below. This notation
    represents a recurrent neural network as a self referential unit.
  </p>
  <!-- usual rnn -->
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 100 150">
      <Arrow
        strokeWidth="1.2"
        data={[
          { x: 40, y: 150 },
          { x: 40, y: 100 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="1.2"
        data={[
          { x: 50, y: 55 },
          { x: 50, y: 10 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="1.2"
        data={[
          { x: 70, y: 75 },
          { x: 90, y: 75 },
          { x: 90, y: 115 },
          { x: 60, y: 115 },
          { x: 60, y: 100 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Block x="50" y="75" width="30" height="30" class="fill-slate-500" />
    </svg>
  </SvgContainer>
  <p>
    As the neural network has to remember the output from the previous run, you
    can say that it posesses a type of a memory. Such a unit is therefore often
    called a <Highlight>memory cell</Highlight> or simply <Highlight
      >cell</Highlight
    >.
  </p>
  <p>
    So far we used only two numbers as an input into a RNN: the current sequence
    value and the previous output. In reality this cell works with vectors just
    like a feedforward neural network. Below for example the unit takes four
    inputs: two come from the part of a sequence and two from the previous
    output.
  </p>

  <!-- several inputs -->
  <SvgContainer maxWidth={"300px"}>
    <svg viewBox="0 0 100 150">
      <Arrow
        strokeWidth="1"
        data={[
          { x: 40, y: 150 },
          { x: 40, y: 100 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="1"
        data={[
          { x: 50, y: 55 },
          { x: 50, y: 10 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Arrow
        strokeWidth="1"
        data={[
          { x: 70, y: 75 },
          { x: 90, y: 75 },
          { x: 90, y: 115 },
          { x: 60, y: 115 },
          { x: 60, y: 100 },
        ]}
        dashed={true}
        moving={true}
        strokeDashArray="4 4"
      />
      <Block x="50" y="75" width="30" height="30" class="fill-slate-500" />

      <!-- 4 inner blocks -->
      <Block x="39" y="85" width="5" height="5" class="fill-blue-200" />
      <Block x="46" y="85" width="5" height="5" class="fill-blue-200" />
      <Block x="53" y="85" width="5" height="5" class="fill-red-400" />
      <Block x="60" y="85" width="5" height="5" class="fill-red-400" />

      <!-- 2 output blocks -->
      <Block x="70" y="75" width="5" height="5" class="fill-red-400" />
      <Block x="78" y="75" width="5" height="5" class="fill-red-400" />

      <!-- 2 input sequence blocks -->
      <Block x="40" y="130" width="5" height="5" class="fill-blue-200" />
      <Block x="40" y="140" width="5" height="5" class="fill-blue-200" />
    </svg>
  </SvgContainer>

  <p>
    We can unroll the recurrent neural network through time. Taking the example
    with a four part sequence from before, the unrolled network will look as
    follows.
  </p>
  <!-- unfolded rnn -->
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
              { x: 100, y: 62 },
              { x: 100, y: 140 },
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
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    While the unrolled network looks like it consists of four units, you
    shouldn't forget that we are dealing with the same layer. That means that
    each of the boxes in the middle has the same weights and the same bias. At
    this point we also make a distinction between outputs <Latex
      >{String.raw`\mathbf{y_t}`}</Latex
    >
    and the hidden units <Latex>{String.raw`\mathbf{h_t}`}</Latex>. For the time
    being there is no difference between the hidden units and the outputs, but
    we will see shortly that there might be differences.
  </p>
  <p>
    We use two sets of weights to calculate the hidden value <Latex
      >{String.raw`\mathbf{h}_t`}</Latex
    >: the weight to process the previous hidden values <Latex
      >{String.raw`\mathbf{W_h}`}</Latex
    > and the weights to process the sequence <Latex
      >{String.raw`\mathbf{W_x}`}</Latex
    >. The hidden value is therefore calculated as <Latex
      >{String.raw`\mathbf{h_t} = f(\mathbf{h_{t-1}}\mathbf{W_h}^T + \mathbf{x_t} \mathbf{W_x}^T + b)`}</Latex
    >. The activation function that is used most commonly with recurrent neural
    networks is tanh. Because we use the very same weights for the whole
    sequence, if the weights are above 1, we will deal with exploding gradients,
    therefore a saturating activation function is preferred. On the other hand a
    long sequence like a sentence or a book, that can consist of hundreds of
    steps, will cause vanishing gradients. We will look into ways of dealing
    with those in the next sections.
  </p>
  <p>
    We will not go over the whole process of backpropagation for recurrent
    neural networks, called <Highlight>backpropagation through time</Highlight>.
    Still we will give you an intuition how you might approach calculating
    gradients for a RNN. In essence backpropagation for fully connected neural
    networks and RNNs is not different. We can use automatic differentiation the
    same way we did in the previous chapters. When you unroll a recurrent neural
    network, each part of a sequence is processed by the same weights and
    gradients are accumulated for those weights in the process. Once the whole
    sequence is exhausted, we can use backpropagation and apply gradient
    descent.
  </p>

  <p>
    Often we will want to create several recurrent layers. In that case the
    hidden outputs of the series are used as the inputs into the next layer.
  </p>
  <!-- multilayer unfolded rnn  -->
  <SvgContainer maxWidth={"350px"}>
    <svg viewBox="0 0 300 470">
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
              { x: 160, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 205, y: 45 },
              { x: 260, y: 45 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 100, y: 62 },
              { x: 100, y: 140 },
            ]}
            dashed={true}
            moving={true}
            strokeDashArray="4 4"
          />
          <Arrow
            strokeWidth="2"
            data={[
              { x: 185, y: 62 },
              { x: 185, y: 140 },
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
          <Block x="185" y="45" width="30" height="30" class="fill-red-500" />
          <Block
            type="latex"
            text="y_{idx + 1}"
            fontSize={12}
            x="285"
            y="45"
            width="25"
            height="25"
            class="fill-blue-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="100"
            y="100"
            width="25"
            height="25"
            class="fill-yellow-100"
          />
          <Block
            type="latex"
            text="h_{idx + 1}"
            fontSize={12}
            x="185"
            y="100"
            width="25"
            height="25"
            class="fill-green-100"
          />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>
    This time around there is a destinction between the output and the hidden
    values. We regard the output to be the hidden values from the very last
    layer.
  </p>
  <p>
    Now let's have a look at what a recurrent neural network does under the hood
    in PyTorch. We will first use <code>nn.RNN</code> on a dummy dataset and then
    manually implement a recurrent neural network using only matrix multiplications.
    This will give us the necessary intuition to work with more complex architectures
    in the future.
  </p>
  <PythonCode
    code={`import torch
import torch.nn as nn`}
  />
  <p>
    Similar to linear and convolutional layers, a recurrent neural network is a <code
      >nn.Module</code
    >. The module takes a parameter <code>nonlinearity</code> as input, which
    determines which activation function is going to be used throughout the
    network. We will stick with the default value of <code>tanh</code>. Below we
    initialize additional parameters that will be useful in our example. We use
    distinct values, in order to be able to differentiate the dimensionalities
    of tensors. If you work through the example below and ask yourself why the
    output is shaped in a particular way, return to these parameters.
  </p>
  <PythonCode
    code={`# number of samples in our dataset
batch_size=4
#sequence lengths represents for example the number of words in a sentence 
sequence_length=5
# dimensionality of each input in the sequence
# so each value in the sequence is a vector of length 6
input_size=6
# the output dimension of each RNN layer
hidden_size=3
# number of recurrent layers in the network
num_layers=2
`}
  />
  <p>
    A recurrent neural network in PyTorch uses an input of shape of <code
      >(sequence length, batch size, features)</code
    >
    as the default. If you set the parameter <code>batch_first</code> to True,
    then you must provide the rnn with the shape
    <code>(batch size, sequence length, features)</code>. For now we will use
    the default behaviour, but in some future examples it will be convenient to
    set this to True.
  </p>
  <p>
    If you want to read more on the module, we refer you to the official PyTorch
    <a
      target="_blank"
      rel="noreferrer"
      href="https://pytorch.org/docs/stable/generated/torch.nn.RNN.html"
      >documentation</a
    >.
  </p>
  <p>
    We create a module and generate two tensors: the first is our dummy data and
    the second is the initial value for the hidden unit.
  </p>
  <PythonCode
    code={`rnn = nn.RNN(input_size=input_size, 
             hidden_size=hidden_size, 
             num_layers=num_layers)`}
  />
  <PythonCode
    code={`sequence = torch.randn(sequence_length, batch_size, input_size)
h_0 = torch.zeros(num_layers, batch_size, hidden_size)`}
  />
  <p>
    The network generates two outputs. The <code>output</code> tensor contains
    the <Latex>{String.raw`\mathbf{y}`}</Latex> values. We get an output of size
    3 for each of the 5 values in the sequence. Given that we use a batch size of
    4, the output dimension is (5, 4, 3). The <code>h_n</code> tensor contains
    the last hidden values for all layers. This would correspond to <Latex
      >{String.raw`\mathbf{h}_5`}</Latex
    > units of size 3, one for each layer and batch.
  </p>
  <PythonCode
    code={`with torch.inference_mode():
    output, h_n = rnn(sequence, h_0)`}
  />
  <PythonCode code={`output.shape`} />
  <PythonCode code={`torch.Size([5, 4, 3])`} isOutput={true} />
  <PythonCode code={`h_n.shape`} />
  <PythonCode code={`torch.Size([2, 4, 3])`} isOutput={true} />
  <p>
    In order to be able to reconstruct the functionality of the <code
      >nn.RNN</code
    >
    module, we will extract the weights and biases that the module was initialized
    with. There are two sets of weights for each of the layer:
    <code>ih</code> is input-hidden set of weights and <code>hh</code> is
    hidden-hidden set of weights. The layers are marked with either
    <code>l0</code>
    or <code>l1</code>. If we created a three layer network, there would be a
    <code>l2</code>.
  </p>
  <PythonCode
    code={`# ---------------------------- #
# layer 1
# ---------------------------- #

# input to hidden weights and biases
w_ih_l0 = rnn.weight_ih_l0
b_ih_l0 = rnn.bias_ih_l0

# hidden to hidden weights and biases
w_hh_l0 = rnn.weight_hh_l0
b_hh_l0 = rnn.bias_hh_l0

# ---------------------------- #
# layer 2
# ---------------------------- #
# input to hidden weights and biases
w_ih_l1 = rnn.weight_ih_l1
b_ih_l1 = rnn.bias_ih_l1

# hidden to hidden weights and biases
w_hh_l1 = rnn.weight_hh_l1
b_hh_l1 = rnn.bias_hh_l1`}
  />
  <p>
    We iterate over the sequence and use the same set of weights and biases per
    layer.
  </p>
  <PythonCode
    code={`def manual_rnn():
    hidden = h_0.clone()
    output = torch.zeros(sequence_length, batch_size, hidden_size)
    with torch.inference_mode():
        for idx, seq in enumerate(sequence):
            for layer in range(num_layers):
                if layer == 0:
                    hidden[0] = torch.tanh(seq @ w_ih_l0.T + b_ih_l0 + hidden[0] @ w_hh_l0.T + b_hh_l0)
                elif layer == 1:
                    hidden[1] = torch.tanh(hidden[0] @ w_ih_l1.T + b_ih_l1 + hidden[1] @ w_hh_l1.T + b_hh_l1)
                    output[idx] = hidden[1]
    return output, hidden
`}
  />
  <p>
    Lastly we compare the outputs of the <code>nn.RNN</code> module and our manual
    implementation. They are mostly identical. The tiny differences are due to rounding
    errors.
  </p>
  <PythonCode code={`manual_output, manual_h_n = manual_rnn()`} />
  <PythonCode code={`torch.sum((output - manual_output) > 0.000001).item()`} />
  <PythonCode code={`0`} isOutput={true} />
  <PythonCode code={`torch.sum((h_n - manual_h_n) > 0.000001).item()`} />
  <PythonCode code={`0`} isOutput={true} />
  <p>
    Now it is time to work through a slighly more useful example. We are going
    to implement a recurrent neural network to predict the sine wave. Our
    implementation is actually based on one of the examples from the official
    PyTorch <a
      href="https://github.com/pytorch/examples"
      target="_blank"
      rel="noreferrer">repo</a
    >. We modified many parts, but we borrowed the general idea.
  </p>
  <PythonCode
    code={`import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)`}
  />
  <p>
    We create 10,000 sine waves of length 50. Each wave is offset by a slighly
    different random number.
  </p>
  <PythonCode
    code={`# params for training
epochs = 100
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# params for sine wave
num_sampples = 10000
seq_len = 50

# creating a sine wave
seq = np.linspace(0, 10, seq_len)
offset = np.random.randint(-5, 5, size=(num_sampples, 1))
data = seq + offset
data = np.sin(data)`}
  />
  <PythonCode
    code={`plt.style.use('bmh')
plt.title('Sine Waves', fontsize=20)
for img in data[:4]:
    plt.plot(img)
    plt.savefig('sine.png')`}
  />
  <img src={sine} alt="four offset sine waves" />
  <p>
    This time around we create our model not using <code>nn.RNN</code>, but the
    <code>nn.RNNCell</code> modules. The cell is basically a single layer of a
    recurrent neural network. If you want to process a sequence by a layer using
    a cell, you iterate over your sequence and process the values one by one.
    You can use both, but often <code>nn.RNN</code> is more convenient.
  </p>
  <p>
    We create two layers. The first layer loops over sequential values from a
    sine function and generates a hidden vector of size 50 at each step. This
    vectors are processed by a second recurrent layer. Finally we use a simple
    linear layer with a tanh activation function. A tanh is convenient for our
    task, as the sine is scaled between -1 and 1 and the tanh squishes values
    into a range between -1 and 1.
  </p>
  <PythonCode
    code={`class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.rnn1 = nn.RNNCell(1, 50)
        self.rnn2 = nn.RNNCell(50, 50)
        self.linear = nn.Linear(50, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_1 = torch.zeros(input.size(0), 50).to(device)
        h_2 = torch.zeros(input.size(0), 50).to(device)

        for input_t in input.split(1, dim=1):
            h_1 = self.rnn1(input_t, h_1)
            h_2 = self.rnn2(h_1, h_2)
            output = self.linear(h_2)
            output = torch.tanh(output)
            outputs += [output]
            
        # predictions used during inference
        with torch.inference_mode():
            for i in range(future):
                h_1 = self.rnn1(output, h_1)
                h_2 = self.rnn2(h_1, h_2)
                output = self.linear(h_2)
                output = torch.tanh(output)
                outputs += [output]
                
        outputs = torch.cat(outputs, dim=1)
        return outputs`}
  />
  <p>
    Our training data consists of all datapoints, with the exception of the last
    one, as we use the last point as our target. We keep back 3 sine waves for
    testing purposes.
  </p>
  <PythonCode
    code={`X_train = torch.from_numpy(data[3:, :-1]).float()
y_train = torch.from_numpy(data[3:, 1:]).float()
X_test = torch.from_numpy(data[:3, :-1]).float()
y_test = torch.from_numpy(data[:3, 1:]).float()`}
  />
  <PythonCode
    code={`class TrainDataset(Dataset):
    def __init__(self):
        self.X = X_train
        self.y = y_train
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]`}
  />
  <PythonCode
    code={`dataset = TrainDataset()
train_dataloader = DataLoader(dataset=dataset, 
                              batch_size=1024, 
                              num_workers=4, 
                              shuffle=True, 
                              drop_last=True)`}
  />
  <PythonCode
    code={`model = Sequence().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)`}
  />
  <PythonCode
    code={`for epoch in range(epochs):
    for features, targets in train_dataloader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(features)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()`}
  />
  <p>
    We use the below code to draw the predictions for the sine waves. The dotted
    lines represent the predictions for 30 datapoints in the future. The results
    are actually quite good.
  </p>
  <PythonCode
    code={`with torch.inference_mode():
    future = 30
    pred = model(X_test.to(device), future=future)
    y = pred.cpu().detach().numpy()

# draw the result
plt.figure(figsize=(15,10))
plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
def draw(yi, color):
    plt.plot(np.arange(X_test.size(1)), yi[:X_test.size(1)], color, linewidth = 2.0)
    plt.plot(np.arange(X_test.size(1), X_test.size(1) + future), yi[X_test.size(1):], color + ':', linewidth = 2.0)
draw(y[0], 'r')
draw(y[1], 'g')
draw(y[2], 'b')
plt.savefig('rnn.png')
plt.close()`}
  />
  <img src={rnn} alt="Prediction of sine waves" />
  <div class="separator" />
</Container>
