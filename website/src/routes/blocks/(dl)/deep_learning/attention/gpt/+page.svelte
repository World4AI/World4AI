<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

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
   <p></p>
   <p>Based on the pre-trainig task, it is obvious that GPT can be used for text generation. You provide the model with a starting text and the model generates word after word to finish your writing. This is where GPT obviously shines, yet this is not the only task GPT is capable of solving. The first iteration of the GPT model, called GPT-1<InternalLink id={1} type="reference" />, was designed to be used for fine-tuning after the model was pretrained on a huge corpus of text. So GPT-1 can be adapted for tasks like summarization.</p>
   <p>Yet the real magic started to happen, when OpenAI released GPT-2<InternalLink type="reference" id={2}/>. The name of the paper "Language Models are Unsupervised Multitask Learners" describes very well what the magic of GPT-2 is. No fine-tuning is required, you use GPT in a zero-shot fashion. That means that the model is able to perform tasks, that it was not trained on without any additional fine-tuning.</p>
</Container>
<Footer {references} />

