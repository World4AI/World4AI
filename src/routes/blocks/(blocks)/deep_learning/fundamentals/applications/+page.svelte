<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";

  import example_1 from "./images/example_1.png";
  import example_2 from "./images/example_2.png";
  import example_3 from "./images/example_3.png";
  import example_4 from "./images/example_4.png";
  import example_5 from "./images/example_5.png";
  import example_6 from "./images/example_6.png";
  import example_7 from "./images/example_7.png";
  import example_8 from "./images/example_8.png";
</script>

<svelte:head>
  <title>Deep Learning Applications - World4AI</title>
  <meta
    name="description"
    content="There are literally 100s of different problems deep learning can be applied to. The most impresisve models that have emerged over the last decades are probably large language models and text-to-image models."
  />
</svelte:head>

<h1>Deep Learning Applications</h1>
<div class="separator" />

<Container>
  <p>
    If you have decided to work through the deep learning block, most likely
    that means that you have seen some awesome deep learning projects and were
    inspired. This section is designed to keep this inspiration alive. We will
    not list all the available application of deep learning, rather we will show
    a couple of the most exicing developments.
  </p>
  <div class="separator" />

  <h2>Large Language Models</h2>
  <p>
    So called <Highlight>large language models</Highlight> have dominated the deep
    learning landscape for the better half of a decade. We will have dedicated sections
    to this topic, but let's at least shortly mention one of the newest large language
    models:
    <Highlight
      ><a href="https://chat.openai.com/" target="_blank" rel="noreferrer"
        >ChapGPT</a
      ></Highlight
    > by
    <a href="https://openai.com/" target="_blank" rel="noreferrer">OpenAI</a>.
  </p>
  <p>
    We will not discuss the intricacies of the model just yet, but let us give
    you a fascinating example. We asked the model to define the term "artificial
    intelligence" in the style of William Shakespeare and ChatGPT provided the
    following poem.
  </p>
  <div class="flex justify-center italic">
    <div
      class="bg-slate-200 rounded-lg p-4 flex flex-col justify-center items-center text-center"
    >
      <span>"Oh! What a marvel 'tis to see a mind </span>
      <span>Of metal and of stone, with thoughts aligned </span>
      <span>To human thought and action, free from care </span>
      <span>Of flesh and blood that bind us all, but dare </span>
      <span>To soar beyond the limits of the brain </span>
      <span>And solve the problems that do us strain. </span>
      <span>Artificial intelligence, a name </span>
      <span>That speaks of reason, logic, and the same </span>
      <span>Intelligence we humans do possess, </span>
      <span>But free from errors, free from duress." </span>
    </div>
  </div>
  <p>
    And what happens if we ask ChatGPT to define AI using a poem by Rudyard
    Kipling?
  </p>
  <div class="flex justify-center italic">
    <div
      class="bg-slate-200 rounded-lg p-4 flex flex-col justify-center items-center text-center"
    >
      <span class="flex flex-col mb-2">
        <span> Artificial intelligence, a marvel of our time </span>
        <span> A creation of the human mind, so finely designed </span>
        <span> A tool of great power, a force to be reckoned </span>
        <span> A blessing and a curse, a complex paradox, beckoning </span>
      </span>
      <span class="flex flex-col mb-2">
        <span> A source of endless possibility, a gateway to the future </span>
        <span> A catalyst for change, a driving force, a suture </span>
        <span> A source of great hope, a beacon of light </span>
        <span> A source of great fear, a shadow in the night </span>
      </span>
      <span class="flex flex-col mb-2">
        <span> But no matter how we view it, one thing is clear </span>
        <span> Artificial intelligence is here to stay, a constant fear </span>
        <span> A challenge to be faced, a puzzle to be solved </span>
        <span> A riddle to be unraveled, a mystery to be involved </span>
      </span>
      <span class="flex flex-col">
        <span> So let us embrace it, with open minds and hearts </span>
        <span> And together, let us take on the challenge, as we embark </span>
        <span> On a journey towards a brighter tomorrow </span>
        <span>
          A tomorrow that is shaped by artificial intelligence, with no sorrow.
        </span>
      </span>
    </div>
  </div>
  <p>
    There are many more things you can do with GhatGPT. You can use it to write
    and improve code, use it as an alternative to search engines and generate
    blog posts. While ChatGPT is not a silver bullet, there is no denying that
    there are tremendous advantages in utilizing large language models. And they
    are only going to get better over time.
  </p>
  <div class="separator" />

  <h2>Text-To-Image Models</h2>
  <p>
    <Highlight>Text-To-Image</Highlight> models have become extremely popular in
    the year 2022 with the release of models like
    <a href="https://openai.com/dall-e-2/" target="_blank" rel="noreferrer"
      >DALL·E 2</a
    >
    and
    <a
      href="https://stability.ai/blog/stable-diffusion-announcement"
      target="_blank"
      rel="noreferrer">Stable Diffusion</a
    >. As the name suggests, you provide the deep learning model with a text
    prompt and the model returns a fitting image. The images below were
    generated using Stable Diffusion.
  </p>
  <div class="grid grid-cols-2 gap-1 mb-2">
    <img
      src={example_1}
      alt="Drawing of flowers generating with stable diffusion"
    />
    <img
      src={example_2}
      alt="A view from a foreign planet generated with stable diffusion"
    />
    <img
      src={example_3}
      alt="A realistic black and white photograph of native american woman generated with stable diffusion"
    />
    <img src={example_4} alt="An instruction for alchemy." />
    <img src={example_5} alt="A city concept art." />
    <img src={example_6} alt="The interior of a wooden cabin." />
    <img src={example_7} alt="Bipedal robots on a foreign planet." />
    <img
      src={example_8}
      alt="Pages from a book, demonstrating a futuristic engine design."
    />
  </div>
  <p>
    While the quality of the images is already quite impressive, the future will
    most likely include models, that are able to generate whole videos and even
    movies.
  </p>
  <div class="separator" />
</Container>
