<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import StepButton from "$lib/button/StepButton.svelte" 

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
        author: "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia",
        title: "Attention is All you Need",
        journal: "Advances in Neural Information Processing Systems",
        year: "2017",
        pages: "",
        volume: "30",
        issue: "",
    }
  ]
  
  const sentence = ["what", "is", "your", "name"];
  let showRnnMovement = false;
  let showTransformerMovement = false;
</script>

<h1>Transformer</h1>
<div class="separator" />

<Container>
  <p>inspired by https://jalammar.github.io/illustrated-transformer/</p>
  <p>Most deep learning researchers and practitioners know the importance of the year 2017. In this year one of the most seminal papers in natural language processing was released by a team at Google<InternalLink type="reference" id={1} />. The title of the paper was "Attention is All you Need". We can start to guess some of the contents of the paper without actually reading the paper. At the time of release most state of the art nlp models used recurrent neural networks with attention mechanisms. The authors argued that recurrent neural networks are actually unnecessary, attention on the other hand is key.
  <p>But why was it so important to get rid of recurrence? Think about how recurrent neural networks work. In order for the model to process the next token in a sequence, it has to have access to the hidden state that is based on all the previous tokens. The tokens in a sequence are processed one at a time. That makes it really hard to parallelize the computations on the GPU. </p>
  <ButtonContainer>
    <StepButton on:click={() => showRnnMovement = !showRnnMovement} />
  </ButtonContainer>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 150"> 
      {#each sentence as word, idx}
        <Block x={70 + 120*idx} y={130} width={90} height={30} fontSize={20} text={word} color="var(--main-color-2)" />
        <Block x={70 + 120*idx} y={30} width={50} height={50} fontSize={20} text={"RNN"} color="var(--main-color-3)" />
        {#if showRnnMovement}
          <Arrow data={[{x:70 + 120*idx, y: 110}, {x: 70 + 120*idx, y:60}]}  showMarker={false} drawLine={true} inDrawParams={{delay: idx*1000}}/>
          <Arrow data={[{x: -20 + 120*idx, y: 30}, {x: 40 + 120*idx, y:30}]} showMarker={false} drawLine={true} inDrawParams={{delay: idx*1000}} />
        {/if}
      {/each}
    </svg>
  </SvgContainer>
  <p>Vaswani et. al. introduced a so called <Highlight>Transformer</Highlight>. The transformer is based on feedforward neural networks and is able to process a whole sequence at a time.</p>

  <ButtonContainer>
    <StepButton on:click={() => showTransformerMovement = !showTransformerMovement} />
  </ButtonContainer>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 250"> 
      {#each sentence as word, idx}
        <Block x={70 + 120*idx} y={220} width={90} height={30} fontSize={20} text={word} color="var(--main-color-2)" />
        {#if showTransformerMovement}
          <Arrow data={[{x:70 + 120*idx, y: 200}, {x: 70 + 120*idx, y:150}]}  showMarker={false} drawLine={true} inDrawParams={{delay: 0}}/>
          <Arrow data={[{x:70 + 120*idx, y: 90}, {x: 70 + 120*idx, y:30}]}  showMarker={false} drawLine={true} inDrawParams={{delay: 1000}}/>
        {/if}
      {/each}
      <Block x={250} y={120} width={450} height={50} fontSize={20} text={"Transformer"} color="var(--main-color-3)" />
    </svg>
  </SvgContainer>

  <p>Transformers have taken the world by storm after their initial release in 2017. Starting with NLP first and slowly but surely spilling into computer vision, reinforcement learning and so on. Nowadays transformers dominate most deep learning research and are an integral part of most state of the art models.</p>
  <p>In our explanations we will closely follow the structure the original paper and we suggest that you attempt to work through it on you own. We can not recomment the paper and the contained illustrations highly enough.</p>
  <div class="separator" />

  <h2>Encoder and Decoder</h2>
  <p>The original transformer architecture was designed for language translation. Similar to recurrent seq-to-seq models it is structured as an encoder-decoder architecture. The encoder takes the (embedded) sentence from the source language and encodes the sentence in parallel. The produced values are passed to the decoder, which in turn produces a tranlated version of the input sentence.</p>
  <SvgContainer  maxWidth="500px">
    <svg viewBox="0 0 400 600">
      <Block x={330} y={30} width={100}  height={20} text={"Softmax"} fontSize={15} color="none" />
      <Block x={330} y={100} width={100}  height={20} text={"Linear"} fontSize={15} color="none" />
      <Block x={330} y={295} width={100}  height={250} text={"Decoder"} fontSize={20} color="var(--main-color-2)" />
      <Block x={70} y={330} width={100}  height={180} text={"Encoder"} fontSize={20} color="var(--main-color-1)" />
      <Block x={330} y={485} width={100}  height={20} text={"Embedding"} fontSize={15} color="none" />
      <Block x={70} y={485} width={100}  height={20} text={"Embedding"} fontSize={15} color="none" />
      <Block x={330} y={580} width={100}  height={20} text={"Output Text"} fontSize={15} color="none" />
      <Block x={70} y={580} width={100}  height={20} text={"Input Text"} fontSize={15} color="none" />

      <Block x={250} y={295} width={30}  height={30} text={"Nx"} fontSize={20} color="var(--main-color-2)" />
      <Block x={150} y={330} width={30}  height={30} text={"Nx"} fontSize={20} color="var(--main-color-1)" />

      <Arrow data={[{x: 70, y: 565}, {x: 70, y: 505}]} strokeWidth=2 />
      <Arrow data={[{x: 330, y: 565}, {x: 330, y: 505}]} strokeWidth=2 />
      <Arrow data={[{x: 70, y: 470}, {x: 70, y: 430}]} strokeWidth=2 />
      <Arrow data={[{x: 330, y: 470}, {x: 330, y: 430}]} strokeWidth=2 />
      <Arrow data={[{x: 330, y: 165}, {x: 330, y: 120}]} strokeWidth=2 />
      <Arrow data={[{x: 330, y: 80}, {x: 330, y: 50}]} strokeWidth=2 />
      <Arrow data={[{x: 70, y: 230}, {x: 70, y: 200}, {x: 270, y: 200}]} strokeWidth=2 />
    </svg>
  </SvgContainer>
  <p>The source text and the output text are embedded by their individual embedding layers, before they are transferred to the encoder and decoder respectively. We depict the encoder slightly smaller, due to a somewhat more complex nature of the decoder, but the components of the encoder and the decoder are actually almost identical. The Nx to the right of the encoder and to the left of the decoder indicate that both blocks are actually made up of several stacked layers. In the original paper 6 encoder and 6 decoder layers were utilized.</p>
  <div class="separator" />

  <h2>Embeddings and Positional Encodings</h2>
  <p>When we use a recurrent net, the relative position of the word in a sentence is implicitly conveyed to the network, because the words are processed in an ordered fashion. A transformer on the other hand processes all words in a sentence at the same time, without caring for the relative position of the word. Yet the order in which a word appears in a sentence does matter for the meaning of that sentece. The embeddings that the transformer requires are therefore more complex, than those we used for a recurrent neural network, because we need to inject addtioinal positional information into those embeddings.</p>
  <p>Positional encodings have to have the same dimensionalities as the values that are produced by the embedding layer, because the embeddings and the positional encoddings are added up to produce the final embeddings, that are used as input into the encoder and decoder.</p>
  <p>The authors of the transformer paper used sine and cosine functions for positional encodings. At the same time they also mentioned that a different approach would be to use a separate embedding layer for positional encodings. We will use the embedding approach in our practical implementation, but you are welcome to implement your own approach. To use an embedding to encode the position is fairly easy. You define an embedding layer which has as many embeddings, as the maximal sentence lengths requires. If you expect the longest sentence to consist of 100 tokens, you will need to encode 100 values. The first token in the sentence will get an embedding that corresponds to index 0, the second word the embedding that corresponds to index 1 and so on. The output of the token embedding and the positional embedding is a 512 dimensional vector. We add both values to get our final embedding.</p>
  <div class="separator" />

  <h2>Attention</h2>
  <p>The type of attention that the transformer uses is called <Highlight>self-attention</Highlight>. Given a sequence of tokens, each token focuses on all parts of the sequence at the same time (including itself), but with different levels of attention, called attention weights.</p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 300"> 
      {#each ["what", "is", "your", "date", "of", "birth", "?"] as word, idx}
        <Block x={50} y={25+idx*40} width={90} height={30} fontSize={20} text={word} color="var(--main-color-2)" />
        <Block x={450} y={25+idx*40} width={90} height={30} fontSize={20} text={word} color="var(--main-color-3)"/>
      {/each}
      {#each ["what", "is", "your", "date", "of", "birth", "?"] as _, idx1}
        {#each ["what", "is", "your", "date", "of", "birth", "?"] as _, idx2}
          <Arrow data={[{x: 100, y: 25+idx1*40}, {x: 400, y: 25+idx2*40}]} strokeWidth={2} moving={true} speed={50} dashed={true} showMarker={false} strokeDashArray="4, 4"/>
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>Essentially we need to calculate a quadratic table with attention weights from each word towards each word.</p>
  <Latex>{String.raw`
    \def\arraystretch{1.5}
       \begin{array}{c:c:c:c:c:c:c:c:c}
       & what & is & your & date & of & birth & ? \\ \hdashline
       what & w_{11} & w_{12} & w_{13} & w_{14} & w_{15} & w_{16} & w_{17} \\
       is & w_{21} & w_{22} & w_{23} & w_{24} & w_{25} & w_{26} & w_{27} \\
       your & w_{31} & w_{32} & w_{33} & w_{34} & w_{35} & w_{36} & w_{37} \\
       date & w_{41} & w_{42} & w_{43} & w_{44} & w_{45} & w_{46} & w_{47} \\
       of & w_{51} & w_{52} & w_{53} & w_{54} & w_{55} & w_{56} & w_{57} \\
       birth & w_{61} & w_{62} & w_{63} & w_{64} & w_{65} & w_{66} & w_{67} \\
       ? & w_{71} & w_{72} & w_{73} & w_{74} & w_{75} & w_{76} & w_{77} \\
    \end{array}
  `}</Latex>
  <p>The attention value for each word can then be calculated as the weighted sum of all embeddings.</p>
  <Latex>{String.raw`\overline{x}_i = \sum_j w_{ij}x_j`}</Latex>
  <p>Where <Latex>i</Latex> is the index of the word that pays attention, <Latex>j</Latex> is the index of the word that receives attention, <Latex>w</Latex> is the attention weight, <Latex>x</Latex> is the embedding of a token and <Latex>{String.raw`\overline{x}`}</Latex> is the weighted average of all embeddings.</p>
  <p>You will notice that we end up with the same amount of embeddings, that we started with, so if your input sentence consists of 10 embeddings, self-attention also proces 10 embeddings. That is all well and good, but what is the point of self-attention? </p>
  <p class="info">The purpose of self attention is to produce context-aware embeddings.</p>
  <p>The easiest way to explain what that means is to look at so called homonyms. Words that are written the same, but have a different meaning. Let's for example look the meaning of the word date.</p>
  <p class="red text-center">What is your <span class="yellow">date</span> of <span class="yellow">birthday</span>?</p>
  <p class="blue text-center">The <span class="yellow">date</span> is my favourite <span class="yellow">fruit</span>.</p>
  <p>In the first sentence the word date will pay attention to itself, but also to birthday and will incorporate the word date and the information that relates to time into a single vector.</p>
  <p>In the second vector, the word date will pay attention to itself and the word fruit, incorporating the "fruitiness" into the vector of the word date.</p>
  <p>Without the self-attention mechanism we would not be able to differentiate between the two words, because word embeddings produce the same vector for the same word, without incorporating the context that surrounds the word. But attention is obviously also useful for words other than homonyms, because it allows to create an embedding for each word, that is specific to the exact context that the word is surrounded by.</p>
  <p>In practice the attention mechanism in transformers involves several steps, that are inspired by information retrieval systems like database queries, so let's take it one step at a time. </p>
  <p>The notion of a query, key and value might be familiar to you if you have ever dealt with a database. If not, below is a simple example.</p>

  <SvgContainer maxWidth="500px">
    <svg viewBox="0 0 500 300">
      <g transform="translate(200, 60)">
         <Block x={0} y={-40} width={60}  height={30} text="key" fontSize={20} color={"var(--main-color-4)"} />
         <Block x={80} y={-40} width={90}  height={30} text="value" fontSize={20} color={"var(--main-color-4)"} />
        {#each Array(5) as _, idx}
          <Block x={0} y={idx*32} width={60}  height={30} text="key {idx+1}" fontSize={20} color={idx===0 ? "var(--main-color-1)" : "var(--main-color-3)"} />
          <Block x={80} y={idx*32} width={90}  height={30} text="value {idx+1}" fontSize={20} color={idx===0 ? "var(--main-color-1)" : "var(--main-color-3)"}/>
        {/each}
      </g>
      <Block x={250} y={280} width={400}  height={30} text="SELECT value WHERE key='key 1'" fontSize={20} color="var(--main-color-2)" />
      <Arrow data={[{x:50, y:280}, {x:5, y:280}, {x:5, y: 20}, {x: 150, y: 20}]} strokeWidth={2}/>
    </svg>
  </SvgContainer>

  <p>In a classical database it is relatively clear what values you will get back from your query. The value is returned, if the query alligns with the key. If for example we use the query "SELECT value WHERE key='key 1'", we should get value 1 in return.</p>
  <p>When we deal with transformers we can think about a more "fuzzy" database, where we don't get a single value for a query, but a weighted sum of all values in the database. Let's for simplicity assume, that we have only two entries in the database with the following vector based keys.</p>
 <Latex>{String.raw`
  k_1 = 
  \begin{bmatrix}
    1  \\
    0  \\
    1  \\
    0  \\
  \end{bmatrix},
  k_2 = 
  \begin{bmatrix}
     0 \\
     1 \\
     1 \\
     0 \\
  \end{bmatrix}
 `}</Latex> 

 <p>We use the following vector based query.</p>
 <Latex>{String.raw`
 q = 
  \begin{bmatrix}
    1  \\
    0  \\
    1  \\
    0  \\
  \end{bmatrix}
 `}</Latex> 

  <p>We can determine the similarity between the query and each of the keys by calculating the dot product and we end up with the following results.</p>
  <p>
    <Latex>{String.raw`s_1 = q \cdot k_1 = 2`}</Latex>,
    <Latex>{String.raw`s_2 = q \cdot k_2 = 1`}</Latex>
  </p>
  <p>The similarity between the query and the first key is larger than with the second key, because the query and the first key are identical. Nevertheless we can say that the query and the second key are somewhat related.</p>
  <p>The dot product similarity scores are transformed in the next step into attention weights, by using those scores as inputs into a softmax function.</p>
  <p>
    <Latex>{String.raw`w_j = \dfrac{e^{s_j}}{\sum_i e^{s_i}}`}</Latex> 
  </p>
  <p>And finally we use those attention weights to calculate the weighted sum of the values from the database, that correspond to the two keys.</p>
  <Latex>{String.raw`a = \sum_j w_{j}v_j`}</Latex>
  <p>This is the value that you retrieve from the database. A weighted sum of database values, that are aggregated based on the similarity between a query and all the keys in the database.</p>
  <p>Now let us try to figure out the connection between this database analogy and the way the transformer actually works.</p>
  <p>In order to calculate the attention the transformer takes embeddings <Latex>E</Latex> as an input. These can be original embeddings from the embedding layer, or outputs from a previous encoder/decoder layer. These embeddings are used as inputs into three different linear layers (without any activations), producing queries <Latex>Q</Latex>, keys <Latex>K</Latex> and values <Latex>V</Latex> respectively. And those three are then combined the attention <Latex>A</Latex>.</p>

  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 300 200">
      <Block x={20} y={100} width={30}  height={30} text="E" fontSize={20} color="var(--main-color-2)" />
      <Block x={150} y={16} width={30}  height={30} text="Q" fontSize={20} color="var(--main-color-2)" />
      <Block x={150} y={100} width={30}  height={30} text="K" fontSize={20} color="var(--main-color-2)" />
      <Block x={150} y={184} width={30}  height={30} text="V" fontSize={20} color="var(--main-color-2)" />
      <Block x={300-16} y={100} width={30}  height={30} text="A" fontSize={20} color="var(--main-color-2)" />
      <Arrow data={[{x: 45, y: 100}, {x:125, y:16}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
      <Arrow data={[{x: 45, y: 100}, {x:125, y:100}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4" />
      <Arrow data={[{x: 45, y: 100}, {x:125, y:184}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>

      <Arrow data={[{x: 170, y: 16}, {x:260, y:90}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
      <Arrow data={[{x: 170, y: 100}, {x:260, y:100}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
      <Arrow data={[{x: 170, y: 184}, {x:260, y:110}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
    </svg>
  </SvgContainer>
  <p>As the queries, keys and values are all based on the same inputs, we are still dealing with self attention, but the linear layers introduce weights, that make the attention mechanism more powerful. For each of the input tokens we have a query, a key and a value, therefore the size of the sequence does not change.</p>
  <Latex>{String.raw`A = \text{softmax}(\dfrac{QK^T}{\sqrt{d}})V`}</Latex>
  <p>The only variable that is unknown to us is <Latex>{String.raw`d`}</Latex>, the dimension of the key. So if we are dealing with a 64 dim vector embedding, we have to divide the similarity by the root of 64. According to the authors this is done, because if the similarity between two vectors is too strong, the softmax might get into a region with very low gradients. The scaling helps to alleviate that problem. The whole expression above is called <Highlight>scaled dot-product attention</Highlight>.</p>
  <p>There is still one caveat we need to discuss. Instead of calculating a single attention <Latex>A</Latex>, we calculate a so called multihead attention. The embeddings have the dimensionality (batch_size, seq_len, 512). When calculating the query, key and value using a linear layer, instead of keeping the vector length of 512, we reduce the dimensionality to 64 and calculate the attention for the smaller vector. The sublayer that does this procedure is called an <Highlight>attention head</Highlight>. We repeat this procedure 8 times, thereby using 8 heads. For that reason this layer is called, <Highlight>multi-head attention</Highlight>. Each head has its own set of weights for queries, keys and values, which allows the network to attend to different parts of the sequence. At the end the results are concatenated, which leaves us again with a vector lengths of 512 (64*8). As we always keep this vector length, we can stack layers upon layers.</p>

  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 300 200">
      <Block x={20} y={100} width={30}  height={30} text="E" fontSize={20} color="var(--main-color-2)" />
      <Block x={150} y={16} width={20}  height={20} text="Q" fontSize={20} color="var(--main-color-2)" />
      <Block x={160} y={26} width={20}  height={20} text="Q" fontSize={20} color="var(--main-color-2)" />
      <Block x={170} y={36} width={20}  height={20} text="Q" fontSize={20} color="var(--main-color-2)" />

      <Block x={150} y={100} width={20}  height={20} text="K" fontSize={20} color="var(--main-color-2)" />
      <Block x={160} y={110} width={20}  height={20} text="K" fontSize={20} color="var(--main-color-2)" />
      <Block x={170} y={120} width={20}  height={20} text="K" fontSize={20} color="var(--main-color-2)" />

      <Block x={150} y={184} width={20}  height={20} text="V" fontSize={20} color="var(--main-color-2)" />
      <Block x={160} y={174} width={20}  height={20} text="V" fontSize={20} color="var(--main-color-2)" />
      <Block x={170} y={164} width={20}  height={20} text="V" fontSize={20} color="var(--main-color-2)" />

      <Block x={300-16} y={100} width={20}  height={20} text="A" fontSize={20} color="var(--main-color-2)" />
      <Block x={300-16-10} y={110} width={20}  height={20} text="A" fontSize={20} color="var(--main-color-2)" />
      <Block x={300-16-20} y={120} width={20}  height={20} text="A" fontSize={20} color="var(--main-color-2)" />

      <Arrow data={[{x: 45, y: 100}, {x:125, y:16}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
      <Arrow data={[{x: 45, y: 100}, {x:125, y:100}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4" />
      <Arrow data={[{x: 45, y: 100}, {x:125, y:184}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>

      <Arrow data={[{x: 170, y: 16}, {x:260, y:90}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
      <Arrow data={[{x: 170, y: 100}, {x:260, y:100}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
      <Arrow data={[{x: 170, y: 184}, {x:260, y:110}]} strokeWidth=2 moving={true} speed={50} dashed={true} strokeDashArray="4 4"/>
    </svg>
  </SvgContainer>
  <div class="separator" />

  <h2>Position-wise Feed-Forward Networks</h2>
  <div class="separator" />

  <h2>Encoding Layer</h2>
  <div class="separator" />
  
  <h2>Decoding Layer</h2>
  <div class="separator" />


  <h2>Softmax</h2>
  <div class="separator" />

  <h2>Training Details</h2>
  <div class="separator" />

</Container>

<Footer {references} />

<style>
  .debug {
      border: 1px solid black;
    }

  .text-center {
      text-align: center;
    }
</style>
