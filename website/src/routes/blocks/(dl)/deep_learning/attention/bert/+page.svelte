<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte"
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte"

  // imports for the diagrams
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
        author: "Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina",
        title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        year: "2018"
    }
  ]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | BERT</title>
  <meta
    name="description"
    content="BERT, or biderectional encoder representation from transformers, is a pretrained model, that can be fine-tuned, for many nlp specific tasks."
  />
</svelte:head>

<h1>BERT</h1>
<div class="separator" />

<Container>
  <p>When the original transformer paper was released in the year 2017, it was not clear what tremendous impact that architecture would have on deep learning. Slowly but surely researchers from different labs begun to release transformer based architectures and the more time passed, the more areas were conquered by transformers. One of the first papers that garnered a lot of attention was BERT by Google<InternalLink id={1} type="reference"/>. The major contribution of BERT was the ease with which you could take a pretrained BERT model and fine-tune it for your specific task. While there were other attempts at introducing transfer learning into natural language, none of the other approaches could stand the test of time. Nowadays BERT (and its variants) is de facto the standard tool that you use for transfer learning in NLP.</p>
  <div class="separator" />

  <h2>Biderectional Encoder</h2>
  <p>BERT is short for <Highlight>B</Highlight>iderectional <Highlight>E</Highlight>ncoder <Highlight>R</Highlight>epresentation from <Highlight>T</Highlight>ransformers. We can infer from the name, that the model architecture consists solely from a transformer encoder, without a decoder. The descritption "biderectional" means, that unlike a decoder, which can only pay attention in one direction to words that were already generated, the encoder can look at all words contained in the sequence. For that reason BERT is primarily used for tasks, that have access to the full sequence at inference time. For example in a classification task we process the whole sequence in order to classify the sentence. When we want to generate next words in the sequence on the other hand, we only have access to the preceding words and are not allowed to look into the future. We will look at the GPT family of decoder models in future sections, which are better suited for text generation.</p>
  <p>A BERT like encoder creates output tokens T, based on all input embeddings E.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 300 150">
      <Block x={40} y={10} width={50} height={12} text="BERT Family" />
      {#each Array(6) as _, idx}
        <Block x={25+idx*50} y={130} width={20} height={20} text="E_{idx+1}" color="var(--main-color-1)" />
        <Block x={25+idx*50} y={35} width={20} height={20} text="T_{idx+1}" color="var(--main-color-3)"/>
        {#each Array(6) as _, idx2}
          <Arrow data={[{x: 25+idx*50, y: 120}, {x:25+idx2*50, y: 45}]} showMarker={false} dashed={true} moving={true} speed={50} />
        {/each}
      {/each}
    </svg>
  </SvgContainer>

  <p>A GPT like decoder creates output tokens T, based on all current or previous input embeddings E.</p>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 300 150">
      <Block x={40} y={10} width={50} height={12} text="GPT Family" />
      {#each Array(6) as _, idx}
        <Block x={25+idx*50} y={130} width={20} height={20} text="E_{idx+1}" color="var(--main-color-1)" />
        <Block x={25+idx*50} y={35} width={20} height={20} text="T_{idx+1}" color="var(--main-color-3)"/>
        {#each Array(6) as _, idx2}
          {#if idx2 >= idx}
            <Arrow data={[{x: 25+idx*50, y: 120}, {x:25+idx2*50, y: 45}]} showMarker={false} dashed={true} moving={true} speed={50} />
          {/if}
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <div class="separator" />
  
  <h2>Pre-Training</h2>
  <p>BERT was trained on an English corpus using two different tasks: <Highlight>masked language model</Highlight> and <Highlight>next section prediction</Highlight>.</p>
  <p>In a masked language model, we replace some parts of the original sequence randomly with a special [MASK] token and the model has to predict the missing word. Let's for example look at the below language to understand, why this task matters for pretraining.</p>
  <p class="light-blue">today i went to a [MASK] to get my hair done</p>
  <p>While you probably have a couple of options to fill out the masked word, your options are still limited by logic and the rules of the english language. The word "hairdresser" is probably the most likely option, but the word "salon" or even "friend" are also valid options. When a model learns to replace [MASK] by a valid option, that shows that in the very least, the model learned the basic statistics that govern a language. And those statistics are useful not only for the task at hand, but many other tasks that require natural language understanding.</p>
  <p>For the next section prediction task, the model faces two sequences and has to decide if the second sequence is the logical continuation from the first sequence.</p>
  <p>The two sentences below seem to have a logical connection, so we expect the model to return true.</p>
  <span class="blue">1: a man went to [MASK] store</span>
  <span class="blue">2: he bought a gallon [MASK] milk</span>
  <p>The next two sentences on the other hand are unrelated and the model should return false.</p>
  <span class="blue">1: a man went to [MASK] store</span>
  <span class="red">2: penguin [MASK] are flightless birds</span>
  <p>What is important to notice that both those tasks are trained using self-supervised learning. We do not require anyone to collect and label a dataset. That gave transformers such an unprecedented potential for pre-training and fine-tuning.</p>
  <div class="separator" />

  <h2>Fine-Tuning</h2>
  <p>When we have a labeled language dataset, we can use BERT to for fine-tuning. How BERT can be used for fine-tuning depends on the task at hand. For that we need to understand what special tokens BERT uses.</p>
  <p>BERT always returns the same amount of tokens, that where used as input. The first token, CLS is the start token and is used for classification tasks. We can use that token as an input into a classification layer and ignore the rest if a category is all we need.</p>
  <SvgContainer maxWidth={"700px"}>
    <svg viewBox="0 0 350 25">
      {#each ["CLS", "TOK 1", "TOK 2", "SEP", "TOK 1", "TOK 2", "SEP"] as content, idx}
        <Block x={25+idx*50} y={12.5} width={30} height={20} text="{content}" color="var(--main-color-3)"/>
      {/each}
    </svg>
  </SvgContainer>
  <p>The SEP tokens are used to separate different sequences. Tasks like text summarization for example require a question as the first sequence and the context that contains the answer.</p>
  <p>What we defined as TOK are the context-aware embeddings for actual input tokens, that were produced by the tokenizer. Many tasks will actually require all the TOK tokens, and not only the CLS token to be solved. In named-entity recognition for example, the model would look at the output embedding of each token and decide if that embedding correstponds to a person (or organisation etc.) or not.</p>
  <div class="separator" />

  <h2>BERT in Practice</h2>
  <p>BERT is designed for fine-tuning, so it makes no sense to train the model from scratch. Instead we will use the pre-trained BERT weights to solve our task at hand. Nowadays the most efficient way to use BERT and its cousins is to use the <a href="https://huggingface.co/" target="_blank">🤗HuggingFace library</a>. This library contains models, pretrained weights, datasets and much more. We will make heavy use of it in future, and not only for natural language processing.</p>
</Container>
<Footer {references} />
