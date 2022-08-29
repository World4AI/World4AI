<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";

</script>

<svelte:head>
  <title>World4AI | Deep Learning | Performance Tuning</title>
  <meta
    name="description"
    content="The requirements to train computer vision models are relatively high. Oftentimes it takes a lot of time on a modest GPU the model is too large to fit on the video RAM. Performance tuning might therefore be extremely important."
  />
</svelte:head>

<h1>Performance Tuning</h1>
<div class="separator" />

<Container>
  <p>You might have noticed, that it takes quite a long time to train computer vision models or that we were not able to use the full batch size that was utilzed according to the papers. As we move along we will encounter performance bottlenecks from time to time and this is generally a good point to discuss possible improvements.</p>
  <p>Performance tuning is not necessarily related to computer vision , but historically speaking huge computer vision models appeared years before large language models. And whatever we discuss in this section can easily be extended to other modalities.</p>
  <p>This section is going to be highly practical and we will mostly avoid theory. The two concpets that we will mostly focus on are: <Highlight>mixed precison training</Highlight> and <Highlight>training using TPUs</Highlight>.</p>
  <div class="separator" />
</Container>
