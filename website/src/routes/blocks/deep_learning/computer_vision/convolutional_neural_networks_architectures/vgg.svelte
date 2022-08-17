<script>
  import Table from "$lib/Table.svelte";
  import Container from "$lib/Container.svelte";
  import JupyterNB from "$lib/JupyterNB.svelte";
  import notebookUrl from "$notebooks/convolutional_neural_networks/vgg.ipynb";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";

  const url = notebookUrl;
  const fileName = "convolutional_neural_networks\\vgg.ipynb";


   let blockHeader = ["Type", "Kernel Size", "Stride", "Padding", "Feature Maps"];
   let blockData = [
  		["Convolution","3x3", "1", "1", "?"],
  		["BatchNorm2d", "-", "-", "-", "-"],
  		["ReLU", "-", "-", "-", "-"]]

   let fcHeader = ["Type", "Size"];
   let fcData = [
  		["Fully Connected","4096"],
  		["Dropout", "0.5"],
  		["ReLU", "-"],
	     ]

   let header = ["Type", "Input Size", "Kernel Size", "Stride", "Padding", "Feature Maps", "Output Size"];
   let data = [
  		["VGG Block", "224x224x3", "3x3", "1", "1", "64", ""],
  		["VGG Block", "", "3x3", "1", "1", "64", ""],
  		["Max Pooling", "", "2x2", "2", "-", "-", ""],
  		["VGG Block", "", "3x3", "1", "1", "128", ""],
  		["VGG Block", "", "3x3", "1", "1", "128", ""],
  		["Max Pooling", "", "2x2", "2", "-", "-", ""],
  		["VGG Block", "", "3x3", "1", "1", "256", ""],
  		["VGG Block", "", "3x3", "1", "1", "256", ""],
  		["VGG Block", "", "3x3", "1", "1", "256", ""],
  		["Max Pooling", "", "2x2", "2", "-", "-", ""],
  		["VGG Block", "", "3x3", "1", "1", "512", ""],
  		["VGG Block", "", "3x3", "1", "1", "512", ""],
  		["VGG Block", "", "3x3", "1", "1", "512", ""],
  		["Max Pooling", "", "2x2", "2", "-", "-", ""],
  		["VGG Block", "", "3x3", "1", "1", "512", ""],
  		["VGG Block", "", "3x3", "1", "1", "512", ""],
  		["VGG Block", "", "3x3", "1", "1", "512", ""],
  		["Max Pooling", "", "2x2", "2", "-", "-", ""],
  		["FC Block", "4096", "", "", "", "", ""],
  		["FC Block", "4096", "", "", "", "", ""],
  		["Fully Connected", "1000", "", "", "", "", ""],
  		["Softmax", "", "", "", "", "", ""],
	     ]

  let references = [{
      author: "Simonyan, K., & Zisserman, A.",
      title: "Very deep convolutional networks for large-scale image recognition",
      journal: "",
      year: "2014",
      pages: "",
      volume: "",
      issue: "",
  }]
</script>

<svelte:head>
  <title>World4AI | Deep Learning | VGG</title>
  <meta
    name="description"
    content="VGG is at heart a very simple convolutional neural network architecture. It stacks layers of convolutions followed by max pooling. But compared to AlexNet or LeNet-5 this architecture showed that deeper and deeper networks might be necessary to achieve truly impressive results." 
  />
</svelte:head>

<h1>VGG</h1>
<div class="separator" />
<Container>
  <p>The VGG<InternalLink type={"reference"} id={1}/> architecture came from the visual geometry group, a computer vision research lab at Oxford university. The neural network is similar in spirit to LeNet-5 and AlexNet, where convolutional layers are stacked upon each other followed by a pooling layer, but vgg does so with many more layers and many more filters per layer. VGG also introduced a practice that is very common to this day. Unlike AlexNet, VGG does not apply any large filters, but uses only small patches of 3x3. Most modern convolutional networks use only 2x2 or 3x3 filters and VGG was the first network to introduce the practice. This design choice lead to the second place in the 2014 ImageNet object detection challenge.</p>
  <p>The VGG paper discussed networks of varying depth, from 11 layers to 19 layers. We are going to discuss the 16 layer architecture, the so called VGG16 (architecture D in the paper).</p>

<p>One of the greatest advantages of VGG is its repeatablity of calculations. In the 19 layer VGG network, a convolutional operation with the same kernel, stride and padding is repeated 2 to 3 times using the same steps. Only the amount of filters varies. That allows us to create a "convolutional block", that stacks a convolution operation, a batch normalization layer and the ReLU activation function. We can reuse that block over and over again, which is useful in the table we provide below, but also in the PyTorch code we will discuss in a short while. Be aware, that the BatchNorm2d layer was not used in the original VGG paper, but if you omit normalization step, the network will suffer from vanishing gradients.</p>
  <p>The VGG block always looks as follows.</p>
  <Table header={blockHeader} data={blockData} />
  <p>There are also two fully connected layers, that essentially do the same operations: a linear operation followed by dropout followed by the ReLu activation function. We will refernce this thres steps as FC Block below.</p>
  <Table header={fcHeader} data={fcData} />
  <p>The full VGG16 implementation looks as follows.</p>
  <Table {header} {data} />
  <p>The original VGG16 architecture has 138 million of trainable parameters. Unless you have a powerful modern graphics card with a lot of vram, your program will crash. Our PyTorch implementation below runs without problems on the free version of Google Colab, but we had to use a batch size that is much smaller than the value provided in the paper, in order to fit the model and the data into the memory of the graphics card.</p>
</Container>

<JupyterNB {url} {fileName} />
<Footer {references} />
