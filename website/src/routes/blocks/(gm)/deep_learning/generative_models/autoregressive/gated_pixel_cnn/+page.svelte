<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Latex from "$lib/Latex.svelte";

  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";

  import Block from "$lib/diagram/Block.svelte";
  import Plus from "$lib/diagram/Plus.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Circle from "$lib/diagram/Circle.svelte";

  const references = [
    {
        author: " van den Oord, Aäron and Kalchbrenner, Nal and Vinyals, Oriol and Espeholt, Lasse and Graves, Alex and Kavukcuoglu, Koray",
        title: "Conditional Image Generation with PixelCNN Decoders",
        year: "2016",
    }
  ];

  const centerIdx = 4;
  const numPixels = 9;
  const pixelSize = 22;
  const gap = 5;
  let layerCommon = 0;
  let layerMasked = 0;
  let layerVertical = 0;
  let layerHorizontal = 0;

  function fCommon() {
    if (layerCommon === 4){
        layerCommon = 0;
    } else {
          layerCommon += 1;
    }
  }

  function fMasked() {
    if (layerMasked === 4){
        layerMasked = 0;
    } else {
          layerMasked += 1;
    }
  }

  function fVertical() {
    if (layerVertical === 4){
        layerVertical = 0;
    } else {
          layerVertical += 1;
    }
  }

  function fHorizontal() {
    if (layerHorizontal === 4){
        layerHorizontal = 0;
    } else {
          layerHorizontal += 1;
    }
  }

  //assume padding of 1 and kernel size 3
  function colorCommon(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx-centerIdx);
    let colDistance = Math.abs(colIdx-centerIdx);
    if (colIdx === centerIdx && rowIdx === centerIdx) {
        return 'black';
    }else if(rowDistance <= layer && colDistance <= layer) {
        return `hsl(10, ${100/Math.max(rowDistance, colDistance)+10}%, 50%)`;
    } else {
        return 'none';
    }     
  }

  function colorMasked(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx-centerIdx);
    let colDistance = Math.abs(colIdx-centerIdx);
    let beyondMiddleLineRow = numPixels-colIdx;
    if (colIdx === centerIdx && rowIdx === centerIdx) {
        return 'black';
     } else if(rowDistance <= layer 
                && colDistance <= layer
                && rowIdx <= centerIdx
                && rowIdx < beyondMiddleLineRow) {
        return `hsl(10, ${100/Math.max(rowDistance, colDistance)+10}%, 50%)`;
    } else {
        return 'none';
    }     
  }

  //assume padding of 1 and kernel size 3
  function colorVertical(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx-centerIdx);
    let colDistance = Math.abs(colIdx-centerIdx);
    if (colIdx === centerIdx && rowIdx === centerIdx) {
        return 'black';
    }else if(rowDistance <= layer && colDistance <= layer && rowIdx < centerIdx) {
        return `hsl(10, ${100/Math.max(rowDistance, colDistance)+10}%, 50%)`;
    } else {
        return 'none';
    }     
  }

  function colorHorizontal(colIdx, rowIdx, layer) {
    let rowDistance = Math.abs(rowIdx-centerIdx);
    let colDistance = Math.abs(colIdx-centerIdx);
    if (colIdx === centerIdx && rowIdx === centerIdx) {
        return 'black';
    }else if(rowDistance <= layer 
              && colDistance <= layer 
              && colIdx < centerIdx 
              && rowIdx === centerIdx) {
        return `hsl(10, ${100/Math.max(rowDistance, colDistance)+10}%, 50%)`;
    } else {
        return 'none';
    }     
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Gated PixelCNN</title>
  <meta
    name="description"
    content="The gated PixelCNN model was developed by DeepMind to improve the generative quality of the common PixelCNN. The model utilizes two stacks of masked convolutions and a gated architecture to improve the performance."
  />
</svelte:head>

<h1>Gated PixelCNN</h1>
<div class="separator"></div>

<Container>
  <p>In this section we will continue our discussion of autoregressive generative models. Specificylly we will improve the PixelCNN model by introducing gated PixelCNNs<InternalLink id={1} type="reference" />.</p>
  <div class="separator"></div>

  <h2>Vertical and horizontal Stacks</h2>
  <p>There are several reasons why we would like to improve the PixelCNN model from the previous section. The most obvious reason is the computational performance advantage of convolutional networks over recurrent networks due to parallelization. RowLSTM on the other hand produced better quality images. Our goal is therefore to keep the computational performance of PixelCNN, while trying to improve the generative performance.</p>
  <p>Let's start by remembering how a convolutional neural network usually works. The very first layer applies convolutions to a very tight receptive field. If we apply a 3x3 convolution, then the neural network can only look at the immediate surrounding of a particular pixel. But as we stack more and more convolutional layers on top of each other, the receptive field starts to grow.</p>
  <p>In this interactive example we assume all calculations are considered from the perspective of the black pixel, the kernel size is 3x3 and the padding is always 1 in order to keep the size of the image constant.</p>
  <ButtonContainer>
    <PlayButton f={fCommon} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect 
            x={gap+colIdx*(pixelSize+gap)} 
            y={gap+rowIdx*(pixelSize+gap)} 
            width={pixelSize} 
            height={pixelSize} 
            stroke="var(--text-color)"
            fill={colorCommon(colIdx, rowIdx, layerCommon)}
            />
        {/each}
      {/each}
    </svg>
  </SvgContainer>

  <p>Now let's see how the receptive field grows, once we incorporate masked convolutions. There is a less obvious problem that we have to face.</p>
  <ButtonContainer>
    <PlayButton f={fMasked} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect 
            x={gap+colIdx*(pixelSize+gap)} 
            y={gap+rowIdx*(pixelSize+gap)} 
            width={pixelSize} 
            height={pixelSize} 
            stroke="var(--text-color)"
            fill={colorMasked(colIdx, rowIdx, layerMasked)}
            />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>While the receptive field grows, we are left with a blind spot. Many pixels above the black dot are not taken into the account, which will most likely deteriorate the performance.</p>

  <p>To deal with this problem the researchers at DeepMind separated the convolution into two distinct stacks: the vertical stack, which processes the pixels above the black pixel and the horizontal stack, which processes the pixels to the left.</p>
  <p>You can think about the vertical stack as a regular convolution, that can only access the upper half of the image.</p>
  <ButtonContainer>
    <PlayButton f={fVertical} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect 
            x={gap+colIdx*(pixelSize+gap)} 
            y={gap+rowIdx*(pixelSize+gap)} 
            width={pixelSize} 
            height={pixelSize} 
            stroke="var(--text-color)"
            fill={colorVertical(colIdx, rowIdx, layerVertical)}
            />
        {/each}
      {/each}
    </svg>
  </SvgContainer>

  <p>The horizontal stack is a 1d convolution that processes the pixels to the left.</p>
  <ButtonContainer>
    <PlayButton f={fHorizontal} />
  </ButtonContainer>
  <SvgContainer maxWidth="250px">
    <svg viewBox="0 0 250 250">
      {#each Array(numPixels) as _, colIdx}
        {#each Array(numPixels) as _, rowIdx}
          <rect 
            x={gap+colIdx*(pixelSize+gap)} 
            y={gap+rowIdx*(pixelSize+gap)} 
            width={pixelSize} 
            height={pixelSize} 
            stroke="var(--text-color)"
            fill={colorHorizontal(colIdx, rowIdx, layerHorizontal)}
            />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>The combination of both produces the desired output. The model is theoretically able to incorporate information from all previously generated pixels to the left and above.</p>
  <div class="separator"></div>

  <h2>Gated Architecture</h2>
  <p>The gated PixelCNN architecture was developed in order to close the performance gap between the PixelCNN and the RowLSTM. The researcher hypothesised, that the multiplicative units from an LSTM can help the model to learn more complex patterns and introduced similar units to the convolutional layers.</p>
  <SvgContainer maxWidth="700px">
    <svg viewBox="0 0 700 400">
      <!-- left part -->
      <g>
        <Arrow data={[{x: 120, y:400}, {x: 120, y: 245}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 120, y:230}, {x: 190, y: 230}, {x: 190, y: 210}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 120, y:230}, {x: 50, y: 230}, {x: 50, y: 210}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 190, y:160}, {x: 190, y: 130}, {x:150 , y: 130}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 50, y:160}, {x: 50, y: 130}, {x:90, y: 130}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 120, y:130}, {x: 120, y: 10}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
  
        <Circle x=120 y=130 r=15 color="var(--main-color-1)"/>
        <Circle x=50 y=180 r=20 color="var(--main-color-3)"/>
        <Circle x=190 y=180 r=20 color="var(--main-color-3)"/>
        <Block x=120 y=230 width=15 height=15 color="var(--main-color-4)" />
        <Block x={120} y={320} width={80} height={30} text="n \times n" type="latex" fontSize="20" color="var(--main-color-2)" />
  
  
        <Block x={50} y={180} width={60} height={20} text="\tanh" type="latex" fontSize="20" color="var(--main-color-3)" border={false} />
        <Block x={190} y={180} width={60} height={20} text="\sigma" type="latex" fontSize="20" color="var(--main-color-3)" border={false} />
        <Block x={120} y={130} width={20} height={20} text="\times" type="latex" fontSize="20" color="var(--main-color-1)" border={false} />
  
        <Block x={100} y={280} width={30} height={30} text="2p" fontSize="18" color="var(--main-color-1)" border={false} />
        <Block x={60} y={250} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
        <Block x={180} y={250} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
        <Block x={100} y={80} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
      </g>
      <!-- right part -->
      <g transform="translate(370 0)">
        <Arrow data={[{x: 120, y:320}, {x: 120, y: 245}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 120, y:230}, {x: 190, y: 230}, {x: 190, y: 210}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 120, y:230}, {x: 50, y: 230}, {x: 50, y: 210}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 190, y:160}, {x: 190, y: 130}, {x:150 , y: 130}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 50, y:160}, {x: 50, y: 130}, {x:90, y: 130}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
        <Arrow data={[{x: 120, y:130}, {x: 120, y: 30}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
  
        <Circle x=120 y=130 r=15 color="var(--main-color-1)"/>
        <Circle x=50 y=180 r=20 color="var(--main-color-3)"/>
        <Circle x=190 y=180 r=20 color="var(--main-color-3)"/>
        <Block x=120 y=230 width=15 height=15 color="var(--main-color-4)" />
        <Block x={120} y={320} width={80} height={30} text="1 \times n" type="latex" fontSize="20" color="var(--main-color-2)" />
  
  
        <Block x={50} y={180} width={60} height={20} text="\tanh" type="latex" fontSize="20" color="var(--main-color-3)" border={false} />
        <Block x={190} y={180} width={60} height={20} text="\sigma" type="latex" fontSize="20" color="var(--main-color-3)" border={false} />
        <Block x={120} y={130} width={20} height={20} text="\times" type="latex" fontSize="20" color="var(--main-color-1)" border={false} />
  
        <Block x={160} y={280} width={30} height={30} text="2p" fontSize="18" color="var(--main-color-1)" border={false} />
        <Block x={60} y={250} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
        <Block x={180} y={250} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
        <Block x={120} y={40} width={80} height={30} text="1 \times 1" type="latex" fontSize="20" color="var(--main-color-2)" />
      </g>

      <!-- vertical to horizontal connection -->
      <Arrow data={[{x: 120, y: 280}, {x: 460, y: 280}]} strokeWidth=2 />
      <Plus x={490} y={280} radius={15} offset={6} color="var(--main-color-1)"/>
      <Block x={300} y={280} width={80} height={30} text="1 \times 1" type="latex" fontSize="20" color="var(--main-color-2)" />

      <!-- skip connection -->
      <Arrow data={[{x: 670, y:400}, {x: 670, y: 10}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
      <Arrow data={[{x: 660, y: 320}, {x: 540, y:320}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
      <Arrow data={[{x: 530, y:40}, {x: 660, y: 40}]} strokeWidth=2 dashed={true} strokeDashArray="4 4" />
      <Plus x={670} y={40} radius={15} offset={6} color="var(--main-color-1)"/>
      <Block x={655} y={320} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
      <Block x={600} y={25} width={20} height={20} text="p" fontSize="18" color="white" border={false} />
    </svg>
  </SvgContainer>
  <p>Let's start our discussion with the left side of the graph, the vertical stack. The vertical layer receives the output from the previous vertical stack and applies an nxn masked convolution of type 'B', where the mask allows the model to only look at the above pixels. The convolution takes in p feature maps and produces twice that amount as the output. This is done because one half goes into the <Latex>\tanh</Latex> and the other goes into the sigmoid activation <Latex>\sigma</Latex>. We multiply both results positionwise. In essence we can interpret the sigmoid output as a sort of gate, that decides which part of the <Latex>\tanh</Latex> output is allowed to flow and which is closed.</p>
  <p>The right side of the graph is the horizontal stack. The calculation is similar to the left side, but with slightly more complexity. First we process the output from the vertical convolution through a 1x1 convolution and add that result to the output of the horizontal convolution. We can't do the addition the other way around, otherwise the vertical stack would get access to future pixels. Second we use skip connections in the vertical stack in order to facilitate training.</p>
  <p>It is ok if you don't fully grasp all the details of the image above. This calculation should get much clearer once we start to implement it in PyTorch. Return to this drawing when you start working through the code.</p>
  <div class="separator"></div>

  <h2>Conditioning</h2>
  <p>Lastly we would like to condition the model on the label we would like to produce. If we are dealing with MNIST for example, we could use the numbers 0-9 as an additional input to the model, so that it can learn to generate specific numbers on demand.</p>
  <div class="separator"></div>
</Container>

<Footer {references} />

<style>
  svg {
    border: 1px solid black;
  }
</style>
