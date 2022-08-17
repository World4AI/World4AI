<script>
  import Table from "$lib/Table.svelte";
  import Container from "$lib/Container.svelte";
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/alexnet.ipynb";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\alexnet.ipynb";

  let header = ["Type", "Input Size", "Kernel Size", "Stride", "Padding", "Feature Maps", "Output Size"];
  let data = [
  		["Convolution", "224x224x3", "11x11", "4", "2", "96", "55x55x96"],
  		["BatchNorm2d", "-", "-", "-", "-", "-", "-"],
  		["Max Pooling", "55x55x96", "3x3", "2", "-", "-", "27x27x96"],
  		["ReLU", "-", "-", "-", "-", "-", "-"],
  		["Convolution", "27x27x96", "5x5", "1", "2", "256", "27x27x256"],
  		["BatchNorm2d", "-", "-", "-", "-", "-", "-"],
  		["Max Pooling", "27x27x256", "3x3", "2", "-", "-", "13x13x256"],
  		["ReLU", "-", "-", "-", "-", "-", "-"],
  		["Convolution", "13x13x256", "3x3", "1", "1", "384", "13x13x384"],
  		["ReLU", "-", "-", "-", "-", "-", "-"],
  		["Convolution", "13x13x384", "3x3", "1", "1", "384", "13x13x384"],
  		["ReLU", "-", "-", "-", "-", "-", "-"],
  		["Convolution", "13x13x384", "3x3", "1", "1", "256", "13x13x256"],
  		["Max Pooling", "13x13x256", "3x3", "2", "-", "-", "6x6x256"],
  		["ReLU", "-", "-", "-", "-", "-", "-"],
  		["Fully Connected", "9219", "-", "-", "-", "-", "4096"],
  		["Fully Connected", "4096", "-", "-", "-", "-", "1000"],
	     ]

  let references = [{
      author: "Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E",
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      journal: "Advances in Neural Information Processing Systems",
      year: "2012",
      pages: "",
      volume: "25",
      issue: "",
  }]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | AlexNet</title>
  <meta
    name="description"
    content="AlexNet is the convolutional network architecture that initiated the latest AI spring. AlexNet showed for the very first time, that showed that the combination of a large amount of data, computational resources and the advances in neural network architectures can produce state of the art results and outperform any other approaches."
  />
</svelte:head>

<h1>AlexNet</h1>
<div class="separator" />
<Container>
  <p>Similar to LeNet, AlexNet<InternalLink type={"reference"} id={1}/> is named after its creator, Alex Krizhevsky. This cnn architecture won the 2012 ImageNet challenge, the ILSVRC (ImageNet Large Scale Visual Recognition Challenge). The dataset that was used for the challenge was a subset of the ImageNet database and consisted of 1,281,167 training images, 50,000 validation images and 100,000 test images with 1,000 different image categories. For the ImageNet dataset it is common practice to calculate top-1 and top-5 error rates. The model outputs the classes with the five highest probabilities.  For the top-1 error rate the model is successful, if the actual class corresponds to the prediction with the highest probability, while for the top-5 error rate the model is successful if the current class is somewhere in the top-5 predictions. In the 2012 challenge AlexNet achieved a top-5 error rate of 15.3%, while the second best entry achieved only 26.2%. This became known as the ImageNet moment. From that point on, neural networks became mainstream and the current AI spring was started.</p>

  <p>When you look at the architecture below, you will notice, that for the most part this architecture does not differ that much from the LeNet-5 architecture. It's just deeper (more layers), wider (more channels) and uses normalization techniques as well as the ReLU activation function.</p>
  
  
    <Table {header} {data} />
  
  <p>The authors preprocessed the data and used image augmentation in order to facilitate the training. As the original images in the ImageNet dataset are of different size and relatively large, they scaled the images to a 256x256 pixels and took a random patch of 224x224, that was used as the input into the neural network. They also applied some transformations, like horizontal flipping and color adjustment and scaled the input features.</p>
  
  <p>For some intermediate layers, the AlexNet architecture utilized so called local response normalization. This normalization step is not used in practice any more and is considered a historical artifact. We will not cover this step in detail, because we instead utilize the BatchNorm2d layers. The 2d batch normalization differs slightly from the 1d version. The mean and the variance are calculated for each channel per batch and the normalization is applied to the whole channel.</p>
  
  <p>The max pooling layers in AlexNet are also very unusual. Normally the kernel size and the stride are of equivalent size, such that the max value is calculated on non overlapping windows. AlexNet on the other hand utilizes overlapping max pooling. You will hardly find such an implementation in more recent convolutional neural networks.</p>
  <div class="separator" />
</Container>

<JupyterNB {url} {fileName} />
<Footer {references} />
