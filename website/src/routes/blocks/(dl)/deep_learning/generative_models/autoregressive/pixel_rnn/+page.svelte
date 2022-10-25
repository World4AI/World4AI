<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte"
  import ButtonContainer from "$lib/button/ButtonContainer.svelte"
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
  
  import SvgContainer from "$lib/SvgContainer.svelte";
  
  import Block from "$lib/diagram/Block.svelte"; 
  import Arrow from "$lib/diagram/Arrow.svelte"; 
  import Plus from "$lib/diagram/Plus.svelte"; 

  const references = [
    {
        author: "Oord, Aaron and Kalchbrenner, Nal and Kavukcuoglu, Koray",
        title: "Pixel Recurrent Neural Networks",
        year: "2016",
    }
  ]

  const imageLength = 4;
  const pixelSize = 25;
  const padding = 1;
  const padding1d = 1;
  const kernel = 3;
  const kernel1d = 3;

  let activeRow = 0;
  let activeCol = 0;
  
  function f() {
    if (activeCol < 3) {
        activeCol += 1;
    } else if (activeCol===3 && activeRow < 3) {
      activeCol = 0; 
      activeRow += 1;
    } else {
        activeCol = activeRow = 0;
      }
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | PixelRNN</title>
  <meta
    name="description"
    content="PixelRNN is a family of autoregressive generative models. The RowLSTM model combines convolutions and a LSTM network to process one row at a time, while the PixelCNN uses only masked convolutions."
  />
</svelte:head>

<h1>PixelRNN</h1>
<div class="separator"></div>
<Container>
  <p><Highlight>PixelRNN</Highlight><InternalLink id=1 type="reference" /> is a family of autoregressive generative models that came out of DeepMind. The research paper covers alltogether 4 types of models: RowLSTM, Diagonal BiLSTM, Multi-Scale PixelRNN and PixelCNN. In this section we will cover RowLSTM and PixelCNN, while in the next section we will look at an improved architecture called Gated PixelCNN.</p>
  <div class="separator"></div>

  <h2>Masked Convolutions</h2>
  <p>While this architecture is called PixelRNN, convolutional layers are a major part of the calculations. Let's utilize a stylized example of a 4x4 image in order to understand what role convolutions play.</p>

  <SvgContainer maxWidth="120px">
    <svg viewBox="0 0 120 120">
      {#each Array(imageLength) as _, colIdx}
        {#each Array(imageLength) as _, rowIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={2 + rowIdx * (pixelSize + 5)} 
                width={pixelSize} 
                height={pixelSize} 
                fill={"var(--main-color-4)"} 
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>We need to process the image in such a way, that the size (height and width) of the input and the output feature maps are identical. So given a 4x4 image and a 3x3 convolution, we need a padding of 1 on each side of the 2d image.</p>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding*2) as _, colIdx}
        {#each Array(imageLength + padding*2) as _, rowIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={2 + rowIdx * (pixelSize + 5)} 
                width={pixelSize} 
                height={pixelSize} 
                fill={colIdx < padding 
                      || rowIdx < padding 
                      || colIdx >= imageLength + padding 
                      || rowIdx >= imageLength + padding
                ? "var(--main-color-3)" : "var(--main-color-4)"} 
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>This is required, because we want to end up with a probability distribution for each pixel. For each of the 4x4 pixels we end up with a 256-way softmax layer. Each of the 256 numbers represents the probability of the pixel to be one of the 256 integer values between 0 and 255. We can use those probabilities to sample pixel values from the multinomial distribution. So in our example we would start with a greyscale image of shape 1x4x4 and end up with 256x1x4x4.</p>
  <p>Now if you look at the kernel below, as indicated by the red dots, you will hopefully see a problem in this type of calculation. The very first output contains knowledge about future pixels. Autoregressive generative models take in previous pixels to generate future pixels. If previous pixels contain knowledge about the future, that would be considered cheating and while our training process would look great, inference would produce garbage, because the model would not have learned how to generate pixels based solely on the past.</p>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding*2) as _, colIdx}
        {#each Array(imageLength + padding*2) as _, rowIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={2 + rowIdx * (pixelSize + 5)} 
                width={pixelSize} 
                height={pixelSize} 
                fill={colIdx < padding 
                      || rowIdx < padding 
                      || colIdx >= imageLength + padding 
                      || rowIdx >= imageLength + padding
                ? "var(--main-color-3)" : "var(--main-color-4)"} 
                stroke="black"/>
        {/each}
      {/each}
      {#each Array(kernel) as _, colIdx}
        {#each Array(kernel) as _, rowIdx}
          <circle cx={2 + colIdx * (pixelSize + 5) + pixelSize/2} 
                cy={2 + rowIdx * (pixelSize + 5) + pixelSize/2} 
                r={5} 
                fill="var(--main-color-1)" 
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>To deal with this problem, we need to apply a mask to the kernel. Essentially we multiply the kernel values that would access future pixels with zeros, thereby zeroing out the weights of kernel positions that relate to future pixels.</p>
  <ButtonContainer>
    <PlayButton {f} delta={500}/>
  </ButtonContainer>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding*2) as _, colIdx}
        {#each Array(imageLength + padding*2) as _, rowIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={2 + rowIdx * (pixelSize + 5)} 
                width={pixelSize} 
                height={pixelSize} 
                fill={colIdx < padding 
                      || rowIdx < padding 
                      || colIdx >= imageLength + padding 
                      || rowIdx >= imageLength + padding
                ? "var(--main-color-3)" : "var(--main-color-4)"} 
                stroke="black"/>
        {/each}
      {/each}
      <!-- draw kernel -->
      {#each Array(kernel) as _, colIdx}
        {#each Array(kernel) as _, rowIdx}
          <circle cx={2 + (colIdx + activeCol) * (pixelSize + 5) + pixelSize/2} 
                cy={2 + (rowIdx + activeRow) * (pixelSize + 5) + pixelSize/2} 
                r={5} 
                fill={rowIdx < padding || (rowIdx === padding && colIdx < padding) ? "var(--main-color-1)" : "none"}
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>There are two types of kernel masks that are used for PixelRNNs. The above mask is of type 'A'. A type 'A' mask zeroes outs the position in the input feature map that corresponds to the position we would like to produce in the output feature map. In the below example on the other hand we use a mask of type 'B'. A type 'B' mask does not zero out the position in the input feature map that corresponds to the position in the output feature map.</p>
  <ButtonContainer>
    <PlayButton {f} delta={500}/>
  </ButtonContainer>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 180">
      {#each Array(imageLength + padding*2) as _, colIdx}
        {#each Array(imageLength + padding*2) as _, rowIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={2 + rowIdx * (pixelSize + 5)} 
                width={pixelSize} 
                height={pixelSize} 
                fill={colIdx < padding 
                      || rowIdx < padding 
                      || colIdx >= imageLength + padding 
                      || rowIdx >= imageLength + padding
                ? "var(--main-color-3)" : "var(--main-color-4)"} 
                stroke="black"/>
        {/each}
      {/each}
      <!-- draw kernel -->
      {#each Array(kernel) as _, colIdx}
        {#each Array(kernel) as _, rowIdx}
          <circle cx={2 + (colIdx + activeCol) * (pixelSize + 5) + pixelSize/2} 
                cy={2 + (rowIdx + activeRow) * (pixelSize + 5) + pixelSize/2} 
                r={5} 
                fill={rowIdx < padding || (rowIdx === padding && colIdx <= padding) ? "var(--main-color-1)" : "none"}
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>The PixelRNN architecture uses type 'A' masks for the input image, while type 'B' masks are applied to all intermediary results. When we get the actual image as input and try to generate a pixel, we have to use mask 'A' in order to hide the actual pixel the model tries to predict. So when we try to predict the very first pixel, the model can only look at zero padded values. Afterwards the value in the top left position does not contain actual information about the original pixel in that position, so we can use mask 'B' safely.</p>
  <div class="separator"></div>

  <h2>Skip Connections</h2>
  <p>In order to facilitate training, PixelRNN utilizes skip connections, by constructing residual blocks.</p>
  <p>A PixelCNN residual block scales down the number of hidden features from 2h to h, before a masked 'B' convolution is applied. Afterwards the dimension is scaled up again and the original input to the block and the output are summed.</p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 220 220">
      <g transform="translate(-5 -10)">
        <Plus x={30} y={40} radius={10} offset={4}/>
        <Block x=160 y=200 width=100 height=30 text="1x1 Conv" fontSize="20px"/>
        <Block x=160 y=120 width=100 height=30 text="3x3 Conv" fontSize="20px"/>
        <Block x=160 y=40 width=100 height=30 text="1x1 Conv" fontSize="20px"/>
        <Arrow data={[{x: 30, y:220}, {x: 30, y: 60}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 30, y:200}, {x: 100, y: 200}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 160, y:180}, {x: 160, y: 145}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 160, y:100}, {x: 160, y: 65}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 110, y:40}, {x: 50, y: 40}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Block x=70 y=180 width=25 height=25 text="2h" fontSize="15px"/>
        <Block x=70 y=60 width=25 height=25 text="2h" fontSize="15px"/>
        <Block x=140 y=80 width=25 height=25 text="h" fontSize="15px"/>
        <Block x=140 y=160 width=25 height=25 text="h" fontSize="15px"/>
      </g>
    </svg>
  </SvgContainer>
  <p>A RowLSTM residual block works in a similar manner, but instead of using only convolutional layers, we use a specialized LSTM block (RowLSTM is described below).</p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 220 220">
      <g transform="translate(-5 -10)">
        <Plus x={30} y={40} radius={10} offset={4}/>
        <Block x=160 y=200 width=100 height=30 text="LSTM" fontSize="20px"/>
        <Block x=160 y=40 width=100 height=30 text="1x1 Conv" fontSize="20px"/>
        <Arrow data={[{x: 30, y:220}, {x: 30, y: 60}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 30, y:200}, {x: 100, y: 200}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 160, y:180}, {x: 160, y: 65}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Arrow data={[{x: 110, y:40}, {x: 50, y: 40}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Block x=70 y=180 width=25 height=25 text="2h" fontSize="15px"/>
        <Block x=70 y=60 width=25 height=25 text="2h" fontSize="15px"/>
        <Block x=140 y=110 width=25 height=25 text="h" fontSize="15px"/>
      </g>
    </svg>
  </SvgContainer>
  <div class="separator"></div>

  <h2>PixelCNN</h2>
  <p>At this point we have all the ingredients to describe a PixelCNN architecture. We use a 7x7 masked convolutional layer of type 'A' to the input image. The output is followed up by 7 residual blocks, which utilize masked convolutions of type 'B'. The final outputs adjust the number of feature maps to the desired 256 and the softmax nonlinearity is applied.</p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 220 200">
        <Arrow data={[{x: 130, y: 0}, {x: 130, y: 190}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Block x=140 y=40 width=150 height=30 text="7x7 Conv, Type 'A'" fontSize="15px" color="var(--background-color)"/>
        <Block x=140 y=80 width=150 height=30 text="3x3 Conv, Type 'B'" fontSize="15px" color="var(--background-color)"/>
        <Block x=20 y=80 width=25 height=25 text="7x" fontSize="15px" color="var(--background-color)"/>
        <Block x=140 y=120 width=150 height=30 text="1x1 Conv" fontSize="15px" color="var(--background-color)"/>
        <Block x=20 y=120 width=25 height=25 text="2x" fontSize="15px" color="var(--background-color)"/>
        <Block x=140 y=160 width=150 height=30 text="256-way Softmax" fontSize="15px" color="var(--background-color)"/>
    </svg>
  </SvgContainer>
  <div class="separator"></div>

  <h2>RowLSTM</h2>
  <p>A RowLSTM layer is a slightly modified LSTM, which processes one row at a time, but instead of using fully connected networks the layer utilizes 1d convolutions. This layer takes the current row <Latex>{String.raw`x_i`}</Latex> and the outputs from the previous iteration: <Latex>{String.raw`h_{i-1}`}</Latex> and <Latex>{String.raw`c_{i-1}`}</Latex>. The current row <Latex>{String.raw`x_i`}</Latex> and the previous hidden state <Latex>{String.raw`h_{i-1}`}</Latex> are both processed by a 1d convolution layer, but while <Latex>{String.raw`h_{i-1}`}</Latex> is processed by a common convolution the current row is processed by a masked 1d convolution of type 'B'.</p>
  <SvgContainer maxWidth="180px">
    <svg viewBox="0 0 180 50">
      {#each Array(imageLength + padding1d*2) as _, colIdx}
          <rect x={2 + colIdx * (pixelSize + 5)} 
                y={25 - pixelSize/2} 
                width={pixelSize} 
                height={pixelSize} 
                fill={colIdx < padding1d || colIdx >= imageLength + padding1d ? "var(--main-color-3)" : "var(--main-color-4)"} 
                stroke="black"/>

      {/each}
      {#each Array(kernel1d) as _, colIdx}
        <!-- draw kernel -->
        <circle cx={2 + (colIdx + activeCol) * (pixelSize + 5) + pixelSize/2} 
                cy={25} 
                r={5}
                fill={colIdx < 2 ? "var(--main-color-1)" : "none"}
                stroke="black"
                />
                
      {/each}
    </svg>
  </SvgContainer>
  <p>Both convolutions produce tensors of size (B, h*4, W), where B is the batch size, h is the hidden size and W is the width of the image. We produce 4*h sized filters, because an LSTM layer requires four neural networks: <Latex>{String.raw`f, g, o, i`}</Latex>. By calculating those simultaneously, we can parallelize the computations better. Overall the RowLSTM looks as follows.</p>
  <Latex>{String.raw`[\mathbf{o_i, f_i, i_i, g_i}] = \sigma(\mathbf{K}^{ss} \circledast \mathbf{h_{i-1}} + \mathbf{K^{is}} \circledast \mathbf{x_i})`}</Latex>
  <p>We generate the output gate <Latex>{String.raw`o`}</Latex>, the forget gate <Latex>{String.raw`f`}</Latex>, the input gate <Latex>{String.raw`i`}</Latex> and the potential addition to the long term memory <Latex>{String.raw`g`}</Latex>, all for the row <Latex>{String.raw`i`}</Latex>. The notations "ss" and "is" are taken from the paper and are standing for input-state and state-state. The <Latex>\circledast</Latex> represents a convolutional operation with the respective <Latex>{String.raw`\mathbf{K}`}</Latex> kernel. <Latex>{String.raw`\sigma`}</Latex> represents the activation function: sigmoid activation for output, forget and intput gates and tanh for the potential addition to the long-term memory. The rest of the calculation is identical to the usual LSTM.</p>
  <Latex>{String.raw`
    \mathbf{c}_i = \mathbf{f}_i \odot \mathbf{c}_{i-1} + \mathbf{i}_i \odot \mathbf{g}_i  \\
    \mathbf{h}_i = \mathbf{o}_i \odot \tanh(\mathbf{c}_i) \\
  `}</Latex>
  <p>We apply this iterative process until all rows in an image are exhaused.</p> 
  <p>The overall RowLSTM architecture is similar to PixelCNN.</p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 220 200">
        <Arrow data={[{x: 130, y: 0}, {x: 130, y: 190}]} strokeWidth={2} dashed={true} strokeDashArray="4 4" moving={true} speed=80 />
        <Block x=140 y=40 width=150 height=30 text="7x7 Conv, Type 'A'" fontSize="15px" color="var(--background-color)"/>
        <Block x=140 y=80 width=150 height=30 text="RowLSTM" fontSize="15px" color="var(--background-color)"/>
        <Block x=20 y=80 width=25 height=25 text="7x" fontSize="15px" color="var(--background-color)"/>
        <Block x=140 y=120 width=150 height=30 text="1x1 Conv" fontSize="15px" color="var(--background-color)"/>
        <Block x=20 y=120 width=25 height=25 text="2x" fontSize="15px" color="var(--background-color)"/>
        <Block x=140 y=160 width=150 height=30 text="256-way Softmax" fontSize="15px" color="var(--background-color)"/>
    </svg>
  </SvgContainer>
  <p>We assume, that at this point in time you might still have many questions regarding the implementation details. We hope those will become clear once you work through the PyTorch implementation of PixelCNN and RowLSTM.</p>
</Container>
<Footer {references} />
