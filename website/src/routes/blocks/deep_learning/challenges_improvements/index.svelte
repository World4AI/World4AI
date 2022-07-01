<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import { onMount } from "svelte";
  import * as tf from "@tensorflow/tfjs";

  let canvasCollection = [];
  let numImages = 10;
  for (let i = 0; i < numImages; i++) {
    let canvas;
    canvasCollection.push(canvas);
  }
  let images = [];
  const csvUrl =
    "https://raw.githubusercontent.com/World4AI/World4AI-Datasets/main/mnist_test.csv";

  onMount(async () => {
    await run();

    canvasCollection.forEach((canvas, idx) => {
      const ctx = canvas.getContext("2d");
      ctx.putImageData(images[idx], 0, 0);
    });
  });

  async function run() {
    const csvDataset = tf.data.csv(csvUrl, {
      columnConfigs: {
        label: {
          isLabel: true,
        },
      },
    });

    const numOfFeatures = (await csvDataset.columnNames()).length - 1;

    const flattenedDataset = csvDataset
      .map(({ xs, ys }) => {
        return { xs: Object.values(xs), ys: Object.values(ys) };
      })
      .shuffle(10)
      .batch(1)
      .take(numImages);

    await flattenedDataset.forEachAsync((e) => {
      let pixelValues = e.xs.dataSync();
      const arr = new Uint8ClampedArray(4 * 28 * 28);
      for (let i = 0; i < arr.length; i += 4) {
        arr[i + 0] = pixelValues[i / 4];
        arr[i + 1] = pixelValues[i / 4];
        arr[i + 2] = pixelValues[i / 4];
        arr[i + 3] = 255;
      }
      let imageData = new ImageData(arr, 28, 28);
      images.push(imageData);
    });
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Challenges and Improvements</title>
  <meta
    name="description"
    content="Training neural networks can sometimes be a challenging problems. Knowing what types of problems you might encounter and techniques to deal with them is a necessary skill for machine learning engineers."
  />
</svelte:head>

<p class="danger">The whole chapter is very early work in progress!</p>
<h1>Challenges and Improvements</h1>
<div class="separator" />

<Container>
  <p>
    The "circular data" example from the previous chapter was trivial to solve.
    In practice it might not be an easy task to make neural networks learn
    though. You could (and most likely will) experience a plethora of problems,
    which could make the training process a misearble experience, expecially if
    you can't find the source of the problem or don't know how to deal with it.
  </p>
  <p>
    In this chapter we will outline some common pitfalls and discuss approaches
    to deal with those problems. Instead of using dummy examples, like we did in
    all the previous chapters, we will solve (simple) real world problems. It is
    tradition in the deep learning community to start with the <Highlight
      >MNIST</Highlight
    > dataset, which is sometimes called the
    <span class="yellow">Hello World of Deep Learning</span>.
    <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">MNIST</a> (Modified
    National Institute of Standards and Technology database) is a collection of 70.000
    handwritten digits, designed to be used in image classification tasks. The images
    contain numbers ranging from 0 to 9 and are of size 28x28 pixels.
  </p>
  <div class="flex-container">
    {#each canvasCollection as canvas}
      <canvas bind:this={canvas} width={28} height={28} />
    {/each}
  </div>
  <p>
    While some of the improvements we are going to cover are not strictly
    necessary to get good results for MNIST, we will still use this opportunity
    to disscuss those techniques. We will utilze them in one of the future
    chapters.
  </p>
  <div class="separator" />
</Container>

<style>
  canvas {
    margin: 0 1px;
  }

  .flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
  }
</style>
