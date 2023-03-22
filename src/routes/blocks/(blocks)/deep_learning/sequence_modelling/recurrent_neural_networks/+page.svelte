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
    In PyTorch we can either use the
    <a
      target="_blank"
      rel="noreferrer"
      href="https://pytorch.org/docs/stable/generated/torch.nn.RNN.html"
      ><code>nn.RNN</code></a
    >
    module or the
    <a
      target="_blank"
      rel="noreferrer"
      href="https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html"
      ><code>nn.RNNCell</code></a
    >. Both can be used to achive the same goal, but
    <code>nn.RNN</code> unrolls the neural net automatically, while the
    <code>nn.RNNCell</code> module needs to be applied to each part of the
    sequence manually. Often it is more convenient to simply use
    <code>nn.RNN</code>, but some more complex architecures will require us to
    use of <code>nn.RNNCell</code>.
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
      >(sequence length, batch size, input_size)</code
    >
    as the default. If you set the parameter <code>batch_first</code> to True,
    then you must provide the shape
    <code>(batch size, sequence length, input_size)</code>. For now we will use
    the default behaviour, but in some future examples it will be convenient to
    set this to True.
  </p>
  <p>
    We create a module and generate two tensors: the first is our dummy sequence
    and the second is the initial value for the hidden state.
  </p>
  <PythonCode
    code={`rnn = nn.RNN(input_size=input_size, 
             hidden_size=hidden_size, 
             num_layers=num_layers,
             nonlinearity='tanh')`}
  />
  <PythonCode
    code={`sequence = torch.randn(sequence_length, batch_size, input_size)
h_0 = torch.zeros(num_layers, batch_size, hidden_size)`}
  />
  <p>
    The recurrent network generates two outputs. The <code>output</code> tensor
    corresponds to the <Latex>{String.raw`\mathbf{y}`}</Latex> values from the diagrams
    above. We get an output vector of dimension 3 for each of the 5 values in the
    sequence and each of the 4 batches, therefore the output dimension is (5, 4,
    3). The <code>h_n</code>
    tensor contains the last hidden values for all layers. This would correspond
    to <Latex>{String.raw`\mathbf{h}_n`}</Latex> values in the diagram above. Given
    that we have 2 layers, 4 batches and hidden units of dimension 3, the dimensionality
    is (2, 4, 3).
  </p>
  <PythonCode
    code={`with torch.inference_mode():
    output, h_n = rnn(sequence, h_0)
print(output.shape, h_n.shape)`}
  />
  <PythonCode
    code={`torch.Size([5, 4, 3]) torch.Size([2, 4, 3])`}
    isOutput={true}
  />
  <p>
    When we want to have more control over the learning process, we might need
    to resort to <code>nn.RNNCell</code>. Each such cell represents a recurrent
    layer, so if you want to use more leayers, you have to create more cells.
  </p>
  <PythonCode
    code={`cell = nn.RNNCell(input_size=input_size, 
                    hidden_size=hidden_size, 
                    nonlinearity='tanh')`}
  />
  <PythonCode
    code={`sequence = torch.randn(sequence_length, batch_size, input_size)
h_n = torch.zeros(batch_size, hidden_size)`}
  />
  <p>
    This time we loop over the sequence manually, always using the last hidden
    state as the input in the next iteration.
  </p>
  <PythonCode
    code={`with torch.inference_mode():
    for t in range(sequence_length):
        h_n = cell(sequence[t], h_t)
print(h_t.shape)`}
  />
  <PythonCode code={`torch.Size([4, 3])`} isOutput={true} />

  <div class="separator" />
</Container>
