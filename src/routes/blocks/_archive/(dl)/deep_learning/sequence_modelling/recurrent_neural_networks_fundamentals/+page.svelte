<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Circle from "$lib/diagram/Circle.svelte";

  // button
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  // coordinates of the circles in the fc diagram
  let fcCircle1x = 10;
  let fcCircle2x = 10;
  let fcCircle1y = 50;
  let fcCircle2y = 150;

  function simulateFc() {
    fcCircle1x+= 1;      
    fcCircle2x+= 1;      
    if (fcCircle1x >= 130 && fcCircle1y < 100.5){
      fcCircle1y+=0.45;
      fcCircle2y-=0.45;
    }

    if (fcCircle1x > 225){
      fcCircle1y = 100;
      fcCircle2y = 100;
    }

    if (fcCircle1x >=350) {
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
      rnnCircle1x+= 1;      
      rnnCircle2x+= 1;      
  
      //  when you reach a declining arrow, start moving in the y direction as well
      if (rnnCircle1x >= 130 && rnnCircle1y < 100.5){
        rnnCircle1y+=0.45;
        rnnCircle2y-=0.45;
      }
      
      // join the two circles
      if (rnnCircle1x > 225){
        rnnCircle1y = 100;
        rnnCircle2y = 100;
      }

      if (rnnCircle1x >= 250 && loop < loops) {
        turnAround = true; 
      }
      // finished
      if (rnnCircle1x >=350) {
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
        rnnCircle1y+= 1;      
        rnnCircle2y+= 1;      
      } else if (rnnCircle1y >= 190 && rnnCircle1x > 10) {
        // move left
        rnnCircle1x-= 1;      
        rnnCircle2x-= 1;      
      } else if (rnnCircle1x <= 10 && rnnCircle1y > 150) {
        // move top
        rnnCircle1y-= 1;      
        rnnCircle2y-= 1;      
      } else {
        turnAround = false;
        rnnCircle1x = 10;
        rnnCircle2x = 10;
        rnnCircle1y = 50;
        rnnCircle2y = 150;
        loop+=1;
      }
    }
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Recurrent Neural Networks Fundamentals</title>
  <meta
    name="description"
    content="A recurrent neural network is very well suited to process sequential data. The RNN processes one piece at a time and keeps the outputs from the previous parts of the sequence in the memory which is used as an additional input in the next step."
  />
</svelte:head>

<h1>The Fundamentals of Recurrent Neural Networks</h1>
<div class="separator" />

<Container>
  <p>Let's start this section by contrasting and comparing a plain vanilla feed forward neural network with a recurrent neural network. We assume that we are dealing with a single neuron (yellow box) that receives two inputs (blue boxes): either from a previous layer or as the raw inputs.</p>
  <p>The feedforward neural network processes the two inputs and sends the output directly to the next layer. Once the inputs have left the neuron, they are forgotten. There is no residiual memory, that those inputs have been processed.</p>
  <!-- fc diagram -->
  <ButtonContainer>
    <PlayButton f={simulateFc} delta={5}/>
  </ButtonContainer>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 400 200">
      <!-- connections -->
      <Arrow strokeWidth=2 data={[{x: 5, y:50}, {x:65, y: 50}]} />
      <Arrow strokeWidth=2 data={[{x: 5, y:150}, {x:65, y: 150}]} />
      <Arrow strokeWidth=2 data={[{x: 130, y:50}, {x:215, y: 90}]} />
      <Arrow strokeWidth=2 data={[{x: 130, y:150}, {x:215, y: 110}]} />
      <Arrow strokeWidth=2 data={[{x: 280, y:100}, {x:380, y: 100}]} />

      <!-- neurons -->
      <Block x=100 y=50 width=50 height=50 color="var(--main-color-4)" />
      <Block x=100 y=150 width=50 height=50 color="var(--main-color-4)" />
      <Block x=250 y=100 width=50 height=50 color="var(--main-color-3)" />

      <!-- moving data -->
      <Circle x={fcCircle1x} y={fcCircle1y} r=5 color="var(--main-color-1)" />
      <Circle x={fcCircle2x} y={fcCircle2y} r=5 color="var(--main-color-1)" />
    </svg>
  </SvgContainer>

  <p>When the model is dealing with sequences, it should probably remember at least some parts of the previous inputs. The meaning of a sentence for example depends on the understanding of the whole sentence and not a single word. So once a model processes a particular word of a sentence, it should have access to the meaning of the words, that came before.</p> 
  <p>A recurrent neural network (RNN) processes each piece of a sequence at a time. At each time step the neuron takes a part of the sequence and its own output from the previous timestep as input. In the very first time step there is no output from the previous step, so it is common to use 0 instead. 
  <p>Below for example we are dealing with a sequence of size 4. Once the sequence is exhaused, the output is sent to the next processing unit.</p>
  <!-- rnn diagram -->
  <ButtonContainer>
    <PlayButton f={simulateRnn} delta={5}/>
  </ButtonContainer>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 400 250">
      {#each Array(4) as _, idx}
        <Block x={50 + idx*40} y=30 width=20 height=20 text={idx+1} fontSize={15} color={loop === idx ? "var(--main-color-1)" : "var(--main-color-4)"} />
      {/each}
      <g transform="translate(0 50)">
        <!-- connections -->
        <Arrow strokeWidth=2 data={[{x: 5, y:50}, {x:65, y: 50}]} />
        <Arrow strokeWidth=2 data={[{x: 250, y:125}, {x:250, y: 190}, {x: 10, y: 190}, {x: 10, y: 150}, {x: 65, y: 150}]} />
        <Arrow strokeWidth=2 data={[{x: 130, y:50}, {x:215, y: 90}]} />
        <Arrow strokeWidth=2 data={[{x: 130, y:150}, {x:215, y: 110}]} />
        <Arrow strokeWidth=2 data={[{x: 280, y:100}, {x:380, y: 100}]} />
  
        <!-- neurons -->
        <Block x=100 y=50 width=50 height=50 color="var(--main-color-4)" />
        <Block x=100 y=150 width=50 height=50 color="var(--main-color-4)" />
        <Block x=250 y=100 width=50 height=50 color="var(--main-color-3)" />
  
        <!-- moving data -->
        <Circle x={rnnCircle1x} y={rnnCircle1y} r=5 color="var(--main-color-1)" />
        <Circle x={rnnCircle2x} y={rnnCircle2y} r=5 color="var(--main-color-1)" />
      </g>
    </svg>
  </SvgContainer>
  <p>It is customary to represent a recurrent neural network as a self referential unit, as depicted below.</p>
  <!-- usual rnn -->
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 100 150">
      <Arrow strokeWidth=1 data={[{x: 40, y:150}, {x:40, y: 100}]} />
      <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
      <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:90, y: 75}, {x:90, y: 115}, {x:60, y: 115}, {x:60, y: 100}]} />
      <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
    </svg>
  </SvgContainer>
  <p>As the neural network has to remember the output from the previous run, you can say that it posesses a type of a memory. Such a unit is therefore often called a <Highlight>memory cell</Highlight> or simply <Highlight>cell</Highlight>.</p>
  <p>While we have represented a cell as a unit, that takes two inputs and outputs one number, in reality this cell works with vectors just like a feedforward neural network. Below for example the unit takes four inputs: two come from the part of a sequence and two from the previous output.</p>

  <!-- several inputs -->
  <SvgContainer maxWidth={"200px"}>
    <svg viewBox="0 0 100 150">
      <Arrow strokeWidth=1 data={[{x: 40, y:150}, {x:40, y: 100}]} />
      <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
      <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:90, y: 75}, {x:90, y: 115}, {x:60, y: 115}, {x:60, y: 100}]} />
      <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
      
      <!-- 4 inner blocks -->
      <Block x=39 y=85 width=5 height=5 color="var(--main-color-1)" />
      <Block x=46 y=85 width=5 height=5 color="var(--main-color-1)" />
      <Block x=53 y=85 width=5 height=5 color="var(--main-color-1)" />
      <Block x=60 y=85 width=5 height=5 color="var(--main-color-1)" />

      <!-- 2 output blocks -->
      <Block x=70 y=75 width=5 height=5 color="var(--main-color-1)" />
      <Block x=78 y=75 width=5 height=5 color="var(--main-color-1)" />

      <!-- 2 input sequence blocks -->
      <Block x=40 y=130 width=5 height=5 color="var(--main-color-1)" />
      <Block x=40 y=140 width=5 height=5 color="var(--main-color-1)" />
    </svg>
  </SvgContainer>

  <p>We can unroll the recurrent neural network through time. Taking the example with a four part sequence from before, the unrolled network will look as follows.</p>
  <!-- unfolded rnn -->
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 150">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 0)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block text={idx+1} fontSize={12} x=50 y=140 width=15 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=20 width=25 height=15 color="var(--main-color-4)" />
          <Block text="H_{idx+1}" fontSize={12} x=125 y=65 width=25 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>While the unrolled network looks like it consists of four units, you shouldn't forget that we are dealing with the same layer. That means that each of the yellow boxes has the same weights. At this point we also make a distinction between outputs <Latex>{String.raw`\mathbf{y_t}`}</Latex> and the hidden units <Latex>{String.raw`\mathbf{h_t}`}</Latex>. In our case there is not going to be a difference between the hidden units and the outputs, but some books and frameworks make that distinction. Essentially they assume that the output takes the hidden units as input and apply one more linear transformation and a nonlinearity: <Latex>{String.raw`\mathbf{y_t} = a(\mathbf{h_t}\mathbf{W}^T`})</Latex>.</p>
  <p>We use two sets of weights to calculate the hidden value <Latex>{String.raw`\mathbf{h}_t`}</Latex>: the weight to process the previous hidden values <Latex>{String.raw`\mathbf{W_h}`}</Latex> and the weights to process the sequence <Latex>{String.raw`\mathbf{W_x}`}</Latex>. The hidden value is therefore calculated as <Latex>{String.raw`\mathbf{h_t} = a(\mathbf{h_{t-1}}\mathbf{W_h}^T + \mathbf{x_t} \mathbf{W_x}^T + b)`}</Latex>. The activation function that is used most commonly with recurrent neural networks is tanh. Because we use the very same weights for the whole sequence, if the weights are above 1, we will deal with exploding gradients. Therefore a saturating activation function is preferred. On the other hand a long sequence like text, that can consist of hundreds of steps, will cause vanishing gradients. We will look into ways of dealing with those in the next sections.</p>
  
  <p>Often we will want to create several recurrent layers. In that case the hidden outputs of the series are used as the inputs into the next layer.</p>
  <!-- multilayer unfolded rnn  -->
  <SvgContainer maxWidth={"800px"}>
    <svg viewBox="0 0 500 250">
      {#each Array(4) as _, idx}
        <g transform="translate({idx*120 - 20}, 100)">
          <Arrow strokeWidth=1 data={[{x: 50, y:140}, {x:50, y: 100}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:55}, {x:50, y: 10}]} />
          <Arrow strokeWidth=1 data={[{x: 50, y:5}, {x:50, y: -90}]} />
          <Arrow strokeWidth=1 data={[{x: 70, y:75}, {x:140, y: 75}] } />
          <Arrow strokeWidth=1 data={[{x: 70, y:-10}, {x:140, y: -10}] } />
          <Block x=50 y=75 width=30 height=30 color="var(--main-color-3)" />
          <Block x=50 y=-10 width=30 height=30 color="var(--main-color-3)" />
          <Block text={idx+1} fontSize={12} x=50 y=140 width=15 height=15 color="var(--main-color-4)" />
          <Block text="Y_{idx+1}" fontSize={12} x=65 y=-80 width=25 height=15 color="var(--main-color-4)" />
          <Block text="H_1_{idx+1}" fontSize={12} x=125 y=65 width=39 height=15 color="var(--main-color-4)" />
          <Block text="H_2_{idx+1}" fontSize={12} x=125 y=-20 width=39 height=15 color="var(--main-color-4)" />
        </g>
      {/each}
    </svg>
  </SvgContainer>
  <p>This time there is a destinction between the output and the hidden values. We regard the output to be the hidden values from the very last layer.</p>
  <p>We will not go over the whole process of backpropagation for recurrent neural networks, but just give you an intuition how you might approach calculating gradients for a RNN. When you unroll a recurrent neural network, there is not much difference between a feedforward and a recurrent neural network. Each sequence is processed several times by the same weights and gradients are accumulated in the process (autograd does that automatically). You can think about a RNN as a feed forward neural network with many layers with the same weights, where the number of layers is dynamic. Only once the whole sequence is processed, can a gradient descent step be taken.</p>
  <div class="separator" />
</Container>


