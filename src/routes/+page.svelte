<script>
  import Button from "$lib/Button.svelte";
</script>

<svelte:head>
  <title>World4AI | Free and Open Source AI Education</title>
  <meta
    name="description"
    content="World4AI is a free and open source education plattform for all AI related topics. This includes machine learning, reinforcement learning, symbolic ai, neuroscience, mathematics, programming and much more."
  />
</svelte:head>

<main class="container mx-auto flex gap-2 flex-col md:flex-row mt-4 w-full">
  <div
    class="bg-white border drop-shadow-md py-10 md:py-20 w-full md:w-1/2 h-[50vh] flex flex-col justify-around items-center"
  >
    <h1 class="text-2xl md:text-3xl lg:text-6xl xl:text-8xl font-semibold">
      <span class="border-2 border-black p-2">AI</span> Education
    </h1>
    <h2 class="italic text-xl md:text-4xl">Free and Open Source</h2>
    <a
      href="/blocks/introduction"
      class="bg-red-500 hover:bg-red-400 text-white rounded-md text-sm uppercase font-bold py-3 px-4"
      >Get Started</a
    >
  </div>
  <div
    class="bg-white border drop-shadow-md w-full md:w-1/2 h-[50vh] md:relative md:top-52"
  />
</main>

<!--
<main class="mt-1">
  <div class="bg-red-500 py-20">
    <h1 class="text-8xl mb-10 font-semibold">
      <span class="border-4 border-black p-3">AI</span> Education
    </h1>
    <h2 class="italic text-4xl m-0">Free and Open Source</h2>
  </div>
</main>
-->
