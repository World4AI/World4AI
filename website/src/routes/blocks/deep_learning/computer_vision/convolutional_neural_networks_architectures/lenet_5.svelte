<script>
  import Table from "$lib/Table.svelte";
  import Container from "$lib/Container.svelte";
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/lenet_5.ipynb";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\lenet_5.ipynb";

  let header = ["Type", "Input Size", "Output Size", "Kernel Size", "Stride", "Padding"];
  let data = [
  		["Convolution", "28x28x1", "28x28x6", "5x5", "1", "2"],
              	["Average Pooling", "28x28x6", "14x14x6", "2x2", "2", "0"],
              	["Convolution", "14x14x6", "10x10x16", "5x5", "1", "0"],
              	["Average Pooling", "10x10x16", "5x5x16", "2x2", "2", "0"],
              	["Convolution", "5x5x6", "1x1x120", "5x5", "1", "0"],
              	["Fully Connected", "120", "84", "-", "-", "-"],
              	["Fully Connected", "84", "10", "-", "-", "-"],
	     ]

  let references = [{
      author: "Y. Lecun, L. Bottou, Y. Bengio and P. Haffner",
      title: "Gradient-based learning applied to document recognition",
      journal: "Proceedings of the IEEE",
      year: "1998",
      pages: "2278-2324",
      volume: "86",
      issue: "11",
  }]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | LeNet-5</title>
  <meta
    name="description"
    content="LeNet-5 is one of the oldest and at the same time one of most well known convolutional neural networks architectures. This architecture was developed by Yann LeCun in 1998 and is still used even to these days to learn the basics of convolutional neural networks."
  />
</svelte:head>

<h1>LeNet-5</h1>
<div class="separator" />
<Container>
  <p>LeNet-5<InternalLink type={"reference"} id={1}/> is one of the oldest and one of the most well known convolutional neural network architectures. The name LeNet-5 is a reference to the inventor of the network, Yann LeCun. The network came out in 1998 and was applied to the MNIST dataset, therefore we can not expect this architecture to produce state of the art results. Nevertheless this is a good exercise and a great starting point in our study of cnn architectures.</p>

  <p>If you look at the original paper, you might notice some differences with our implementation. For example we use the cross-entropy as the loss function, while the original LeNet-5 architecture uses a so called RBF layer (Euclidian Radial Basis Function). We make these adjustments to make the architecture more recognizable and more similar to modern networks. You will notice when you study older papers, that some implementation details did not stand the test of time and it makes little sense to focus on those details.</p>

  <p>In the table below we present the architecure of the network. Our implementation of LeNet-5 uses 3 convolutional layers and average pooling after the first two layers. After the third convolutional layer we flatten the feature maps and use two fully connected layers. We use the tanh activation function for all convolutional and fully connected layers, only the last fully connected layer uses the softmax activation function. </p>
  <Table {header} {data} />
  <div class="separator" />
</Container>

<JupyterNB {url} {fileName} />
<Footer {references} />
