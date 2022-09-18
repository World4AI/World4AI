<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";

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
</script>

<h1>Transformer</h1>
<div class="separator" />

<Container>
  <p>inspired by https://jalammar.github.io/illustrated-transformer/</p>
  <p>Most deep learning researchers and practitioners know the importance of the year 2017. In this year one of the most seminal papers in natural language processing was released by a team at Google<InternalLink type="reference" id={1} />. The title of the paper was "Attention is All you Need". We can start to guess some of the contents of the paper without actually reading the paper. At the time of release most state of the art nlp models used recurrent neural networks with attention mechanisms. The authors argued that recurrent neural networks are actually unnecessary, attention on the other hand is key.
  <p>But why was it so important to get rid of recurrence? Think about how recurrent neural networks work. In order for the model to process the next token in a sequence, it has to have access to the hidden state that is based on all the previous tokens. The tokens in a sequence are processed one at a time. That makes it really hard to parallelize the computations on the GPU. Vaswani et. al. introduced a so called <Highlight>Transformer</Highlight>. The transformer is based on feedforward neural networks and is able to process a whole sequence at a time.</p>
  <p class="danger">TODO: animations showing the sequential and parallel processing</p>
  <p>Transformers have taken the world by storm after their initial release in 2017. Starting with NLP first and slowly but surely spilling into computer vision, reinforcement learning and so on. Nowadays transformers dominate most deep learning research and are an integral part of most state of the art models.</p>
  <p>In our explanations we will closely follow the structure the original paper and we suggest that you attempt to work through it on you own. We can not recomment the paper and the container illustrations highly enough.</p>
  <div class="separator" />

  <h2>Encoder and Decoder</h2>
  <p>The original transformer architecture was designed for language translation. Similar to recurrent seq-to-seq models it is structured as an encoder-decoder architecture. The encoder takes the (embedded) sentence from the source language and creates a vector for each of the input words. Those values are passed to the decoder, which in turn produces a tranlated version of the input sentence.</p>
  <p class="danger">TODO: show encoder decoder here</p>
  <p class="yellow">The encoder takes a sequence of the original language and generates a context-aware embedding for each token in the input sentence. The classical embeddings that we have encountered in the previous sections provide exactly one embedding per word. It does not matter if we have the sentence "the bank opens at .." or "the bank of the river...", the word bank will get assigned the same embedding. The decoder combines the previously translated part of the target sentence and the produced embeddings from the input sentence and generates the next word in the sequence.</p> 
  <div class="separator" />

  <h2>Attention</h2>
  <p>The type of attention that the transformer uses is called <Highlight>self-attention</Highlight>. Given a sequence of tokens, each token focuses on all parts of the sequence at the same time (including itself), but with different levels of attention.</p>
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

  <Latex>{String.raw`
    \def\arraystretch{1.5}
       \begin{array}{c:c:c:c:c:c:c:c:c}
       & what & is & your & date & of & birth & ? \\ \hdashline
       what & w_{12} & w_{12} & w_{13} & w_{14} & w_{15} & w_{16} & w_{17} \\
       is & w_{21} & w_{22} & w_{23} & w_{24} & w_{25} & w_{26} & w_{27} \\
       your & w_{31} & w_{32} & w_{33} & w_{34} & w_{35} & w_{36} & w_{37} \\
       date & w_{41} & w_{42} & w_{43} & w_{44} & w_{45} & w_{46} & w_{47} \\
       of & w_{51} & w_{52} & w_{53} & w_{54} & w_{55} & w_{56} & w_{57} \\
       birth & w_{61} & w_{62} & w_{63} & w_{64} & w_{65} & w_{66} & w_{67} \\
       ? & w_{71} & w_{72} & w_{73} & w_{74} & w_{75} & w_{76} & w_{77} \\
    \end{array}
  `}</Latex>
  <div class="separator" />

  <h2>Position-wise Feed-Forward Networks</h2>
  <div class="separator" />

  <h2>Encoding Layer</h2>
  <div class="separator" />
  
  <h2>Decoding Layer</h2>
  <div class="separator" />

  <h2>Embeddings</h2>
  <div class="separator" />

  <h2>Positional Encoding</h2>
  <div class="separator" />

  <h2>Softmax</h2>
  <div class="separator" />

  <h2>Training Details</h2>
  <div class="separator" />

</Container>

<Footer {references} />

