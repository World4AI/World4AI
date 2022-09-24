<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
      author: "Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever",
      title: "Improving Language Understanding by Generative Pre-Training",
      journal: "",
      year: "2018",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author: "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever",
      title: "Language Models are Unsupervised Multitask Learners",
      journal: "",
      year: "2019",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author: "Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et. al",
      title: "Language Models are Few-Shot Learners",
      journal: "",
      year: "2020",
      pages: "",
      volume: "",
      issue: "",
    },
  ]

  const sentence = ["what", "is", "your", "name"];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | BERT</title>
  <meta
    name="description"
    content="GPT, short for generative pre-training, is a family of decoder based transformer models developed by OpenAI. The models are trained based on next token predictions and can perform zero-shot, one-shot or few-shots learning."
  />
</svelte:head>

<h1>GPT</h1>
<div class="separator" />

<Container>
  <p>GPT, short for <Highlight>generative pre-training</Highlight>, is a family of models developed by researchers at OpenAI. Unlike BERT, GPT is a decoder based transformer model, without any encoder interaction.</p> 
  <SvgContainer  maxWidth="500px">
     <svg viewBox="0 0 400 350">
        <Block x={200} y={175} width={200}  height={300} text={""} fontSize={20} color="var(--main-color-2)" />
        <Block x={380} y={80} width={30}  height={30} text={"Nx"} fontSize={20} color="var(--main-color-2)" />
  
        <!-- 3 bottom arrows -->
        <Arrow data={[{x: 200, y: 350}, {x: 200, y: 275}]} strokeWidth={2}/>
        <Arrow data={[{x: 200, y: 350}, {x: 200, y: 290}, {x: 140, y: 290}, {x: 140, y: 275}]} strokeWidth={2}/>
        <Arrow data={[{x: 200, y: 350}, {x: 200, y: 290}, {x: 260, y: 290}, {x: 260, y: 275}]} strokeWidth={2}/>
  
        <!-- first skip connection -->
        <Arrow data={[{x: 200, y: 350}, {x: 200, y: 310}, {x: 50, y: 310}, {x: 50, y: 200}, {x: 115, y: 200}]} strokeWidth={2}/>
  
        <!-- connect attention to feed forward -->
        <Arrow data={[{x: 200, y: 250}, {x: 200, y: 125}]} strokeWidth={2}/>
  
        <!-- second skip connection -->
        <Arrow data={[{x: 200, y: 170}, {x: 50, y: 170}, {x: 50, y: 50}, {x: 115, y: 50}]} strokeWidth={2}/>
  
        <!-- connect fc to next layer-->
        <Arrow data={[{x: 200, y: 100}, {x: 200, y: 10}]} strokeWidth={2}/>
  
        <!-- decoder components -->
        <Block x={200} y={50} width={150}  height={30} text={"Add & Norm"} fontSize={15} color="var(--main-color-2)" />
        <Block x={200} y={100} width={150}  height={30} text={"P.w. Feed Forward"} fontSize={15} color="var(--main-color-2)" />
  
        <Block x={200} y={200} width={150}  height={30} text={"Add & Norm"} fontSize={15} color="var(--main-color-2)" />
        <Block x={200} y={250} width={150}  height={30} text={"Multihead Atention"} fontSize={15} color="var(--main-color-2)" />
        <Block x={90} y={250} width={60}  height={30} text={"Masked"} fontSize={15} color="var(--main-color-2)" />
     </svg>
   </SvgContainer>
   <p>The training objective is of GPT is quite simple. Given some tokens from a sentence, predict the next token.</p>
    <SvgContainer maxWidth={"350px"}>
      <svg viewBox="0 0 150 150">
        <Block x={40} y={10} width={50} height={12} text="GPT Family" />
        {#each Array(3) as _, idx}
          <Block x={25+idx*50} y={130} width={30} height={20} text="{sentence[idx]}" color="var(--main-color-1)" />
          <Block x={25+idx*50} y={35} width={30} height={20} text="{sentence[idx+1]}" color="var(--main-color-3)"/>
          {#each Array(3) as _, idx2}
            {#if idx2 >= idx}
              <Arrow data={[{x: 25+idx*50, y: 120}, {x:25+idx2*50, y: 45}]} showMarker={false} dashed={true} moving={true} speed={50} />
            {/if}
          {/each}
        {/each}
      </svg>
    </SvgContainer>
   <p>Based on the pre-trainig task, it is obvious that GPT can be used for text generation. You provide the model with a starting text and the model generates word after word to finish your writing. This is where GPT obviously shines, yet this is not the only task GPT is capable of solving. The first iteration of the GPT model, called GPT-1<InternalLink id={1} type="reference" />, was designed to be used for fine-tuning after the model was pretrained on a huge corpus of text. So GPT-1 can be adapted for tasks like text summarization. Similar to BERT you put a linear layer on top of the output of GPT to fine-tune your model in a supervised fashion.</p>
   <p>Yet the real magic started to happen, when OpenAI released GPT-2<InternalLink type="reference" id={2}/>. The name of the paper "Language Models are Unsupervised Multitask Learners" describes very well what the magic of GPT-2 is. No fine-tuning is required, you use GPT in a zero-shot fashion. That means that the model is able to perform tasks, that it was not trained on without any additional fine-tuning.</p>
   <p>GPT-1 is trained in such a way, that the model tries to maximize the probability of the next token in the sentence, given the previous tokens.</p>
   <Latex>{String.raw`p(output | input)`}</Latex> 
   <p>So if the transformer receives the tokens "what", "is", "your" as input, the token with the highest probability as the output might be "name".</p>
   <p>Researchers at OpenAI tried to create a more general deep learning system through GPT-2, that can generalize to many nlp tasks simultaneously. They argued, that additionally to the usual input of the transformer, you could theoretically provide the task you would want to perform. Based on the task, the model should output different tokens.</p>
   <Latex>{String.raw`p(output | input, task)`}</Latex> 
   <p>For a translation task, you could provide ("what is your name", "translate to german") and the model should output the sentence "wie heißt du".</p>
   <p>But remember that all GPT models are optimized on next token predictions using a single dataset without any additional task specific tokens. So how is it able to generalize to those tasks? Let's take one example from the GPT-2 paper to make that clear.</p>
   <p class="light-blue">”I’m not the cleverest man in the world, but like they say in French: Je ne suis pas un imbecile [I’m not a fool].</p>
   <p>While GPT-2 is generally trained on english text, text passages as above provide context for english-french tranlations. It is not necessary to explicitly provide the task "translation" during training, the model is able to infer that task from the data. There are at least two reasons why this works. The first reason is the size of the model. The original GPT-1 model has 117 million parameters, while GPT-2 has 1,542 million parameters, which is more than 1 order of magnitude increase. This addiitonal parameters can capture the meaning of the text in much granular way and learn to extract the "task" from the language directly. The second reason is the large dataset, that contains 40GB of text data. Such a huge corpus will inevitably contain many examples of translations, summarizations and much more.</p>
   <p>At inference time you actually provide the task, you want GPT-2 to perform. To induce text summarization for an article for example, you have to attach TL;DR: after the article. TLDR stands for "too long, didn't read" in internet lingo. This expression is often used at the beginning or end of internet articles before a short summary of a longer text.</p>
   <p>GPT-3<InternalLink id={3} type="reference" /> continued this trend of scaling up model parameters and the dataset, culminating in a model consisting of 175 billion parameters. While the performace of GPT-3 is impressive and the examples of generated text look surreal, there is a caveat. OpenAI did not release the weights of the model due to ethical concerns. The research lab had the same reservations during the release of GPT-2, but they eventually released the weights after a period of time. Most likely this is not going to be the case for GPT-3 for the forseeable future. At this point in time the underlying  model is exclusively licenced by Microsoft. You can still use the model using the official OpenAI <a href="https://openai.com/api/" target="_blank">api</a> to build your app, but you can not get direct access to the weights. This is not necessarily the worst option, considering that the model is huge and you most likely need special infrastructure. But if you need an open source implementation, luckily there are companies, like <a href="https://www.eleuther.ai/" target="_blank">EleutherAI</a>, that attemt to replicate the OpenAI GPT-3 model. At the moment of writing their largest model, GPT-NeoX-20B consists of 20 billion parameters, but they plan to train a 175 billion parameter model, somewhere in the future.</p>
</Container>
<Footer {references} />

