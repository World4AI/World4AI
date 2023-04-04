<script>
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Sentence from "$lib/BlockLetters/Sentence.svelte";
  import Card from "$lib/Card.svelte";
</script>

<svelte:head>
  <title>Artificial Intelligence Blocks - World4AI</title>
  <meta
    name="description"
    content="World4AI is a organized into different blocks. Each block represents a topic that is relevant to artificial intelligence. For example deep learning, symbolic ai or neurosciece."
  />
</svelte:head>

<article>
  <Container>
    <h1>Blocks</h1>
    <div class="separator" />
    <p>
      World4AI is organized into building blocks. Each block forms knowledge
      that is fundamental to modern artificial intelligence.
    </p>
    <SvgContainer maxWidth={"700px"}>
      <Sentence sentence={"blocks"} />
    </SvgContainer>
    <p>
      We intend to cover the basics of deep learning, symbolic ai, neurosciece
      and much more, but at the moment only a few sections have been started and
      none is considered finalized. More chapters will be published in the next
      weeks and months. But the overall process will most likely take many
      years.
    </p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
      <Card>
        <span class="text-xs inline-block bg-red-400 py-1 px-2 rounded-lg"
          >not implemented yet</span
        >
        <h2>Mathematics</h2>
        <hr />
        <p>
          Mathematics is the foundation for all artificial intelligence. If you
          want to understand AI in depth, you will need to invest some time into
          this block. Be aware though, that you do not need to work through all
          of mathematics in order to start learning AI. You can always come back
          here.
        </p>
      </Card>
      <Card>
        <span class="text-xs inline-block bg-blue-400 py-1 px-2 rounded-lg"
          >ready to use</span
        >
        <h2>Deep Learning</h2>
        <hr />
        <p>
          Most of the recent successes in artificial intelligence is due to deep
          learning. Computer vision, natural language processing, generative
          art, board games and many more areas have been taken over by deep
          neural networks. Yet all those achievements will pale in comparison to
          those that will surely come in the next decades. This is a fascinating
          area to explore.
        </p>
        <p>
          The block is suited for absolute beginners, but exposure to linear
          algebra, probability theory, calculus and Python is assumed.
        </p>
        <hr />
        <div class="flex justify-center">
          <a
            href="/blocks/deep_learning/fundamentals/introduction"
            class="inline-block transition border border-black bg-red-500 hover:bg-white text-white hover:text-black rounded-md py-3 px-12 m-2 font-bold text-center no-underline"
            >Start Learning</a
          >
        </div>
      </Card>

      <Card>
        <span class="text-xs inline-block bg-yellow-200 py-1 px-2 rounded-lg"
          >work in progress</span
        >
        <h2>Reinforcement Learning</h2>
        <hr />
        <p>
          A reasonable way to think about reinforcement learning is to think
          about animals whom you would like to teach good behaviour and
          discourage from bad behaviour. The way you would go about it would
          probably involve some reward when the animal behaves the way you
          desire and punishment when it doesn’t.
        </p>
        <p>
          This block is suited for beginners, but we rely on some fundamental
          knowledge of deep learning for more advanced topics.
        </p>
        <hr />
        <div class="flex justify-center">
          <a
            href="/blocks/reinforcement_learning/intuition/introduction"
            class="inline-block transition border border-black bg-red-500 hover:bg-white text-white hover:text-black rounded-md py-3 px-12 m-2 font-bold text-center no-underline"
            >Start Learning</a
          >
        </div>
      </Card>
      <Card>
        <span class="text-xs inline-block bg-red-400 py-1 px-2 rounded-lg"
          >not implemented yet</span
        >
        <h2>GOFAI</h2>
        <hr />
        <p>
          GOFAI stands for <em>good old fashioned artificial intelligence</em>.
          In this block we indend to cover topics like search and planning.
        </p>
      </Card>
      <Card>
        <span class="text-xs inline-block bg-red-400 py-1 px-2 rounded-lg"
          >not implemented yet</span
        >
        <h2>Neuroscience</h2>
        <hr />
        <p>
          Neuroscience is the study of the human brain. While this topic might
          not be absolutely necessary to understand AI, there are a lot of
          lessons that we can learn about AI by studying the only organ capable
          of complex behaviour, the human brain.
        </p>
      </Card>
    </div>
    <div class="separator" />
  </Container>
</article>
