<script>
  import landing1 from "./_landing_img/landing_1.png";
</script>

<svelte:head>
  <title>World4AI | Free and Open Source AI Education</title>
  <meta
    name="description"
    content="World4AI is a free and open source education plattform for all AI related topics. This includes machine learning, reinforcement learning, symbolic ai, neuroscience, mathematics, programming and much more."
  />
</svelte:head>

<main class="container mx-auto">
  <div class="flex flex-col md:flex-row mt-20 gap-2 text-center">
    <div class="my-auto p-5">
      <h1 class="text-5xl sm:text-6xl xl:text-8xl m-0">Learn AI</h1>
      <h2 class="text-5xl sm:text-6xl xl:text-8xl m-0">From The Ground Up</h2>
      <a
        href="/blocks/introduction"
        class="inline-block transition border hover:border-black bg-red-500 hover:bg-white text-white hover:text-black rounded-md py-3 px-12 m-10 font-bold text-center no-underline"
        >Start Here</a
      >
    </div>
    <div class="mx-auto">
      <img
        src={landing1}
        class="rounded-md"
        alt="Brain made out of tubes and machinery"
      />
    </div>
  </div>
</main>
