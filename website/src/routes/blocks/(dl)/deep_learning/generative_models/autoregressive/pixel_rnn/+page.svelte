<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte"
  import ButtonContainer from "$lib/button/ButtonContainer.svelte"
  import Latex from "$lib/Latex.svelte";
  
  import SvgContainer from "$lib/SvgContainer.svelte";

  const references = [
    {
        author: "Oord, Aaron and Kalchbrenner, Nal and Kavukcuoglu, Koray",
        title: "Pixel Recurrent Neural Networks",
        year: "2016",
    }
  ]

  const imageLength = 4;
  const pixelSize = 25;
  const padding = 3;
  const padding1d = 1;
  const kernel = 7;
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
    content="PixelRNN is an autoregressive generative model, that combines convolutions and a LSTM network to process one row at a time."
  />
</svelte:head>

<h1>PixelRNN</h1>
<div class="separator"></div>
<Container>
  <p>PixelRNN<InternalLink id=1 type="reference" /> is an autoregressive generative model that came out of DeepMind. The research paper covers alltogether 4 types of models: RowLSTM, Diagonal BiLSTM, Multi-Scale PixelRNN and PixelCNN. In this section we will cover a simplified version of RowLSTM, while in the next section we will talk about PixelCNN. The "simplified" version of our RowLSTM implementation relates to the fact, that we will deal with black and white MNIST images, which will simplify some of the calculations. Additionally we will use fewer layers in our implementation, as RNNs are hard to parallelize and the training process is relatively slow, even with a single recurrent layer.</p>
  <p>While this architecture is called PixelRNN, convolution layers are a major part of the calculations. Let's utilize a stylized example in order to understand what role convolutions play.</p>
  <p>We assume that our image is of size {imageLength} by {imageLength}.</p>

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
  <p>We need to process the image in such a way, that the input size and the output size are identical. Given a 7x7 convolution, we need a padding of 3 on each side of the 2d image.</p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 300">
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
  <p>But if you look at the kernel, as indicated by the red dots, you will hopefully see a problem in this type of the calculation. The very first output contains knowledge about future pixels. Autoregressive generative models take in previous pixels to generate future pixels. If previous pixels contain knowledge about the future, that would be considered cheating and while our training process would look great, inference would produce garbage, because the model would not have learned how to generate pixels based solely on the past.</p>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 300">
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
  <p>To deal with this problem, we need to apply a mask to the kernel. Essentially we zero out the weights of kernel positions that relate to future pixels.</p>
  <ButtonContainer>
    <PlayButton {f} delta={500}/>
  </ButtonContainer>
  <SvgContainer maxWidth="300px">
    <svg viewBox="0 0 300 300">
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
                fill={rowIdx < 3 || (rowIdx === 3 && colIdx < 3) ? "var(--main-color-1)" : "none"}
                stroke="black"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>This layer is going to generate several feature maps, 16 in our MNIST implementation.</p>
  <p>After the initial 2d convolution layer, we apply several RowLSTM layers. A RowLSTM layer is a slightly modified LSTM, which processes one row at a time and instead of using fully connected networks, it utilizes 1d convolutions. This layer takes the current row <Latex>{String.raw`x_i`}</Latex> and the outputs from the previous iteration: <Latex>{String.raw`h_{i-1}`}</Latex> and <Latex>{String.raw`c_{i-1}`}</Latex>. The current row <Latex>{String.raw`x_i`}</Latex> and the previous hidden state <Latex>{String.raw`h_{i-1}`}</Latex> are both processed by a 1d convolution layer, but while <Latex>{String.raw`h_{i-1}`}</Latex> is processed by a common convolution the current row is processed by a masked 1d convolution.</p>
  
  <p>The mask that we use in a 1d convolution to process <Latex>{String.raw`x_i`}</Latex> covers future pixels, but unlike the 2d mask, the model can look at the current pixel value.</p>
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
  <p>Both convolutions produce tensors of size (B, H*4, W), where B is the batch size, H is the hidden size and W is the width of the image. We produce 4*H sized filters, because we have <Latex>{String.raw`f, g, o, i`}</Latex> networks in the common LSTM. By calculating those simultaneously, we can parallelize the computations better. Overall the RowLSTM looks as follows.</p>
  <Latex>{String.raw`[\mathbf{o_i, f_i, i_i, g_i}] = \sigma(\mathbf{K}^{ss} \circledast \mathbf{h_{i-1}} + \mathbf{K^{is}} \circledast \mathbf{x_i})`}</Latex>
  <p>Where we generate the output gate <Latex>{String.raw`o`}</Latex>, the forget gate <Latex>{String.raw`f`}</Latex>, the input gate <Latex>{String.raw`i`}</Latex> and the potential addition to the long term memory <Latex>{String.raw`g`}</Latex>, all for the row <Latex>{String.raw`i`}</Latex>. The notations "ss" and "is" are taken from the paper and are standing for input-state and state-state. <Latex>{String.raw`\sigma`}</Latex> represents the sigmoid activation for output, forget and intput gates and tanh for the potential addition to the long-term memory. The rest of the calculation is identical to the usual LSTM.</p>
  <Latex>{String.raw`
    \mathbf{c}_i = \mathbf{f}_i \odot \mathbf{c}_{i-1} + \mathbf{i}_i \odot \mathbf{g}_i  \\
    \mathbf{h}_i = \mathbf{o}_i \odot \tanh(\mathbf{c}_i) \\
  `}</Latex>
  <p>We apply this iterative process until all rows in an image are exhaused. In the original paper the authors applied several such RowLSTM layers and used skip connections, but as this implementation is inherently slow, we will apply just one layer.</p> 
  <p>In the final layers, we apply convolutions with 1d kernels and bring down the number of features to the original value. So if we started out with a 1x28x28 image, we end up with a 1x28x28 image. The output is used as input to the sigmoid function and we compare each pixel value using the negative log likelihood as loss.</p>
  <p>We assume that some implementation details might still be unclear at this point in time, so we suggest you read the original paper and work through our PyTorch implementation.</p>
</Container>
<Footer {references} />
