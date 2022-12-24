<script>
  import { ArrowRight } from "lucide-svelte";
</script>

<svelte:head>
  <title>World4AI | Free and Open Source AI Education</title>
  <meta
    name="description"
    content="World4AI is a free and open source education plattform for all AI related topics. This includes machine learning, reinforcement learning, symbolic ai, neuroscience, mathematics, programming and much more."
  />
</svelte:head>

<main class="polka container mx-auto mt-4 w-full h-[90vh]">
  <div
    class="polka bg-white py-10 md:py-20 w-full flex flex-col justify-between items-center"
  >
    <h1
      class="flex flex-col lg:flex-row text-6xl sm:text-8xl xl:text-8xl font-bold mb-10"
    >
      <span> Intuition.</span>
      <span> Math.</span>
      <span> Code.</span>
    </h1>
    <h2 class="text-lg md:text-2xl font-normal m-10 max-w-3xl text-gray-600">
      World4AI is a free and open source AI education plattform, that takes you
      from an absolute beginner to an intermediary level.
    </h2>
    <a
      href="/blocks/introduction"
      class="flex gap-2 transition border border-red-500 hover:border-black bg-white hover:bg-black hover:text-white rounded-md py-3 px-12 m-10 font-bold text-center"
      ><span>Start Learning</span> <ArrowRight /></a
    >
  </div>
</main>

<style>
  .polka {
    background-image: radial-gradient(#dadada 1px, transparent 1px);
    background-size: 20px 20px;
  }
</style>
