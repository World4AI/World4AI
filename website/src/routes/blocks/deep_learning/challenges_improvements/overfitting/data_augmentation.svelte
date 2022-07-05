<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import { onMount } from "svelte";
  import * as tf from "@tensorflow/tfjs";

  let canvasCollection = [];
  let transformedCanvasCollection = [];
  let numImages = 10;
  for (let i = 0; i < numImages; i++) {
    let canvas;
    canvasCollection.push(canvas);

    let transformedCanvas;
    transformedCanvasCollection.push(transformedCanvas);
  }
  let images = [];
  let transformedImages = [];

  const csvUrl =
    "https://raw.githubusercontent.com/World4AI/World4AI-Datasets/main/mnist_test.csv";

  onMount(async () => {
    await run();

    canvasCollection.forEach((canvas, idx) => {
      const ctx = canvas.getContext("2d");
      ctx.putImageData(images[idx], 0, 0);
    });

    transformedCanvasCollection.forEach((canvas, idx) => {
      const ctx = canvas.getContext("2d");
      ctx.putImageData(transformedImages[idx], 0, 0);
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
      .batch(1)
      .take(numImages);

    await flattenedDataset.forEachAsync((e) => {
      // create original image
      let pixelValues = e.xs.dataSync();
      let arr = new Uint8ClampedArray(4 * 28 * 28);
      for (let i = 0; i < arr.length; i += 4) {
        arr[i + 0] = pixelValues[i / 4];
        arr[i + 1] = pixelValues[i / 4];
        arr[i + 2] = pixelValues[i / 4];
        arr[i + 3] = 255;
      }
      let imageData = new ImageData(arr, 28, 28);
      images.push(imageData);

      // create transformed image

      //let img = e.xs;
      let img = e.xs.reshape([1, 28, 28, 1]);
      img = tf.image.rotateWithOffset(img, 0.5);
      img = img.reshape([1, 784]);
      let transformedPixelValues = img.dataSync();
      arr = new Uint8ClampedArray(4 * 28 * 28);
      for (let i = 0; i < arr.length; i += 4) {
        arr[i + 0] = transformedPixelValues[i / 4];
        arr[i + 1] = transformedPixelValues[i / 4];
        arr[i + 2] = transformedPixelValues[i / 4];
        arr[i + 3] = 255;
      }
      imageData = new ImageData(arr, 28, 28);
      transformedImages.push(imageData);
    });
  }
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Data Augmentation</title>
  <meta
    name="description"
    content="We do not always posess sufficient amounts of data to avoid overfitting. Data augmentation is a simple technique to produce synthetic data that can be used to train a neural network."
  />
</svelte:head>

<h1>Data Augmentation</h1>
<div class="separator" />

<Container>
  <p>
    One of the best ways to reduce the chances of overfitting is to gather more
    data. Lets assume we are dealing with MNIST and want to teach a neural net
    to recognize hand written digits. If we provide the neural network with a
    limited amount of data, there is a very little chance, that the network will
    learn to recognize the digits. Instead it will memorize the specific
    samples. If we provide the network with millions of images, the network has
    no chance to memorize all those images. It will have no choice, but to
    generalize to the high amount of data.
  </p>
  <p>
    MNIST provides 60,000 training images and 10,000 test images. This data is
    sufficient to train a good performing neral network, because the task is
    comparatively easy. In modern day deep learning this amount of data would be
    insufficient and we would be required to collect more data. Oftentimes
    collection of additional samples is not feasable and we will resort to <Highlight
      >data augmentation</Highlight
    >.
  </p>
  <p class="info">
    Data augmentation is a techinque that applies transformations to the
    original dataset, thereby creating synthetic data, that can be used in
    training.
  </p>
  <p>
    We could for example rotate, blur or flip the images, but there are many
    more options available. Below we rotate the MNIST data by half a radian.
  </p>

  <div class="flex-container">
    {#each canvasCollection as canvas}
      <canvas bind:this={canvas} width={28} height={28} />
    {/each}
  </div>
  <div class="flex-container">
    {#each transformedCanvasCollection as canvas}
      <canvas bind:this={canvas} width={28} height={28} />
    {/each}
  </div>
  <p>
    It is not always the case that we would take the 60,000 MNIST training
    example, apply let's say 140,000 transformations and end up with 200,000
    images for training. Often we apply some random transformations to each
    batch of traning that we encounter. For example we could slightly rotate and
    blur each of the 32 images in our batch using some random parameters. That
    way our neural network never encounters the same image twice and has to
    learn to generalize.
  </p>
  <p>
    It is relatively easy to augment image data, but it is not always easy to
    augment text or time series data. To augment text data on Kaggle for
    example, in some competitions people used google translate to translate a
    sentence into a foreign language first and then translate the sentence back
    into english. The sentence changes slightly, but is similar enough to be
    used in the training process. Sometimes you might need to get creative to
    find a good data augmentation approach.
  </p>
  <p>
    Before we move on let us mention that there is a significantly more powerful
    technique to deal with limited data: <Highlight>transfer learning</Highlight
    >. Tranfer learning allows you to use a model, that was pretrained on
    millions of images or millions of texts, thereby allowing you to finetune
    the model to your needs. Those types of models need significantly less data
    to learn a particular task. It makes little sense to cover transfer learning
    in detail, before we have learned convolutional neural networks or
    transformers. Once we encounter those types of networks we will discuss the
    topic in more detail.
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
    margin-bottom: 5px;
  }
</style>
