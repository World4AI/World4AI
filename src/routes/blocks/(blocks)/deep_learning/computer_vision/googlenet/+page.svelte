<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Convolution from "../../computer_vision/_convolution/Convolution.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  //diagram
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  let header = ["Type", "Input Size", "Output Size"];
  let data = [
    ["Basic Block", "224x224x3", "112x112x64"],
    ["Max Pooling", "112x112x64", "56x56x64"],
    ["Basic Block", "56x56x64", "56x56x64"],
    ["Basic Block", "56x56x64", "56x56x192"],
    ["Max Pooling", "56x56x192", "28x28x192"],
    ["Inception", "28x28x192", "28x28x256"],
    ["Inception", "28x28x256", "28x28x480"],
    ["Max Pooling", "28x28x480", "14x14x480"],
    ["Inception", "14x14x480", "14x14x512"],
    ["Inception", "14x14x512", "14x14x512"],
    ["Inception", "14x14x512", "14x14x512"],
    ["Inception", "14x14x512", "14x14x528"],
    ["Inception", "14x14x528", "14x14x832"],
    ["Max Pooling", "14x14x832", "7x7x832"],
    ["Inception", "7x7x832", "7x7x832"],
    ["Inception", "7x7x832", "7x7x1024"],
    ["Avg. Pooling", "7x7x1024", "1x1x1024"],
    ["Dropout", "-", "-"],
    ["Fully Connected", "1024", "1000"],
    ["Softmax", "1000", "1000"],
  ];

  let references = [
    {
      author:
        "Szegedy, Christian and Wei Liu and Yangqing Jia and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew",
      title: "Going deeper with convolutions",
      journal:
        "2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      year: "2015",
      pages: "1-9",
      volume: "",
      issue: "",
    },
    {
      author: "Lin, M., Chen, Q., & Yan, S.",
      title: "Network in Network",
      journal: "",
      year: "2013",
      pages: "",
      volume: "",
      issue: "",
    },
    {
      author:
        "Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, ZB",
      title: "Rethinking the Inception Architecture for Computer Vision",
      journal: "",
      year: "2016",
      pages: "",
      volume: "",
      issue: "",
    },
  ];

  let gap = 1;

  // basic block
  let basicWidth = 90;
  let basicHeight = 100;
  let basicBoxWidth = 80;
  let basicBoxHeight = 20;
  let basicMaxWidth = "200px";

  //create diagram for inception block
  let width = 1000;
  let height = 500;
  let boxWidth = 210;
  let boxHeight = 70;
  let maxWidth = "800px";

  const components = [];
</script>

<svelte:head>
  <title>GoogLeNet - World4Ai</title>
  <meta
    name="description"
    content="The GoogLeNet architecture combines several layers of Inception modules to create a deep convolutional neural network. An inception module simultaneously calculates convolutions with different kernel sizes using the same input and the results are then concatenated."
  />
</svelte:head>

<h1>GoogLeNet</h1>
<div class="separator" />
<Container>
  <p>
    The <Highlight>GoogLeNet</Highlight><InternalLink
      type={"reference"}
      id={1}
    /> architecture was developed by researchers at Google, but the name is also
    a reference to the original LeNet-5 architecture, a sign of respect for Yann
    LeCun. GoogLeNet achieved a top-5 error rate of 6.67% (VGG achieved 7.32%) and
    won the 2014 ImageNet classification challenge.
  </p>

  <p>
    The GoogLeNet network is a specific, 22 layer, realization of the so called
    Inception architecture. This architecture uses an Inception block, a
    multibranch block that applies convolutions of different filter sizes to the
    same input and concatenates the results in the final step. This architecture
    choice removes the need to search for the optimal patch size and allows the
    creation of much deeper neural networks, while being very efficient at the
    same time. In fact the GoogLeNet architecture uses 12x fewer parameters than
    AlexNet.
  </p>
  <p>
    In the very first step we create a basic building block that is going to be
    utilized in each convolutional layer. The block constists of a convolutional
    layer with variable filter and feature map size. The convolution is followed
    by a batch norm layer and a ReLU activation function. In the original
    implementation batch normalization was not used, instead in order to deal
    with vanishing gradients, the authors implemented several losses along the
    path of the neural network. This approach is very uncommon and we are not
    going to implement these so called auxilary losses. Batch normalization is a
    much simpler and practical approach.
  </p>
  <SvgContainer maxWidth={basicMaxWidth}>
    <svg viewBox="0 0 {basicWidth} {basicHeight}">
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="Conv2d"
      />

      <Block
        x={basicWidth / 2}
        y={basicHeight / 2}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="BatchNorm2d"
      />
      <Block
        x={basicWidth / 2}
        y={basicBoxHeight / 2 + gap}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="ReLU"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - gap,
          },
          {
            x: basicWidth / 2,
            y: basicHeight / 2 + basicBoxHeight / 2 + 3,
          },
        ]}
        dashed={true}
        moving={true}
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight / 2 - basicBoxHeight / 2,
          },
          {
            x: basicWidth / 2,
            y: basicBoxHeight + 4,
          },
        ]}
        dashed={true}
        moving={true}
      />
    </svg>
  </SvgContainer>
  <p>
    The Inception block takes an input from a previous layer and applies
    calculations in 4 different branches using the basic block from above. At
    the end the four branches are concatenated into a single output.
  </p>
  <SvgContainer {maxWidth}>
    <svg viewBox="0 0 {width} {height}">
      <!-- First Layer -->
      <Block
        x={width / 2}
        y={boxHeight / 2 + gap}
        width={boxWidth}
        height={boxHeight}
        text="Concatenation"
        fontSize={25}
      />
      <Block
        x={boxWidth / 2 + gap}
        y={height - boxHeight - height / 2}
        width={boxWidth}
        height={boxHeight}
        text="1x1 Basic Block"
        fontSize={25}
      />
      <Block
        x={(width / 3) * 1}
        y={height - boxHeight - height / 2}
        width={boxWidth}
        height={boxHeight}
        text="3x3 Basic Block"
        fontSize={25}
      />
      <Block
        x={(width / 3) * 2}
        y={height - boxHeight - height / 2}
        width={boxWidth}
        height={boxHeight}
        text="5x5 Basic Block"
        fontSize={25}
      />
      <Block
        x={(width / 3) * 3 - boxWidth / 2 - gap}
        y={height - boxHeight - height / 2}
        width={boxWidth}
        height={boxHeight}
        text="1x1 Basic Block"
        fontSize={25}
      />
      <!-- second layer -->
      <Block
        x={(width / 3) * 1}
        y={height - height / 3}
        width={boxWidth}
        height={boxHeight}
        text="1x1 Basic Block"
        fontSize={25}
      />
      <Block
        x={(width / 3) * 2}
        y={height - height / 3}
        width={boxWidth}
        height={boxHeight}
        text="1x1 Basic Block"
        fontSize={25}
      />
      <Block
        x={(width / 3) * 3 - boxWidth / 2 - gap}
        y={height - height / 3}
        width={boxWidth}
        height={boxHeight}
        text="3x3 MaxPool"
        fontSize={25}
      />
      <!-- input -->
      <Block
        x={width / 2}
        y={height - boxHeight / 2 - gap}
        width={boxWidth}
        height={boxHeight}
        text="Input"
        fontSize={25}
      />

      <!-- arrows -->
      <Arrow
        data={[
          {
            x: width / 2,
            y: height - boxHeight,
          },
          {
            x: (width / 3) * 1,
            y: height - height / 3 + boxHeight / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: width / 2,
            y: height - boxHeight,
          },
          {
            x: (width / 3) * 2,
            y: height - height / 3 + boxHeight / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />

      <Arrow
        data={[
          {
            x: width / 2 + boxWidth / 2,
            y: height - boxHeight / 2,
          },
          {
            x: (width / 3) * 3 - boxWidth / 2 - gap,
            y: height - boxHeight / 2,
          },
          {
            x: (width / 3) * 3 - boxWidth / 2 - gap,
            y: height - height / 3 + boxHeight / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: width / 2 - boxWidth / 2,
            y: height - boxHeight / 2,
          },
          {
            x: 0 + boxWidth / 2 + gap,
            y: height - boxHeight / 2,
          },
          {
            x: 0 + boxWidth / 2 + gap,
            y: height - boxHeight / 2 - height / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: (width / 3) * 1,
            y: height - height / 3 - boxHeight / 2,
          },
          {
            x: (width / 3) * 1,
            y: height - boxHeight / 2 - height / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: (width / 3) * 2,
            y: height - height / 3 - boxHeight / 2,
          },
          {
            x: (width / 3) * 2,
            y: height - boxHeight / 2 - height / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: (width / 3) * 3 - boxWidth / 2 - gap,
            y: height - height / 3 - boxHeight / 2,
          },
          {
            x: (width / 3) * 3 - boxWidth / 2 - gap,
            y: height - boxHeight / 2 - height / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: 0 + boxWidth / 2 + gap,
            y: height - boxHeight - boxHeight / 2 - height / 2,
          },
          {
            x: 0 + boxWidth / 2 + gap,
            y: boxHeight / 2,
          },
          {
            x: width / 2 - boxWidth / 2 - 10,
            y: boxHeight / 2,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: (width / 3) * 1,
            y: height - boxHeight - boxHeight / 2 - height / 2,
          },
          {
            x: width / 2 - boxWidth / 2,
            y: boxHeight / 2 + boxHeight / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: (width / 3) * 2,
            y: height - boxHeight - boxHeight / 2 - height / 2,
          },
          {
            x: width / 2 + boxWidth / 2,
            y: boxHeight / 2 + boxHeight / 2 + 10,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
      <Arrow
        data={[
          {
            x: (width / 3) * 3 - boxWidth / 2 - gap,
            y: height - boxHeight - boxHeight / 2 - height / 2,
          },
          {
            x: (width / 3) * 3 - boxWidth / 2 - gap,
            y: boxHeight / 2,
          },
          {
            x: width / 2 + boxWidth / 2 + 10,
            y: boxHeight / 2,
          },
        ]}
        moving={true}
        strokeWidth={3}
        dashed={true}
        strokeDashArray="14 4"
        speed={50}
      />
    </svg>
  </SvgContainer>
  <!--
  <Diagram arrowStrokeWidth={3} {width} {height} {maxWidth} {components} />
  -->
  <p>
    You will notice that aside from the expected 3x3 convolutions, 5x5
    convolutions and max pooling, there is a 1x1 convolution in each single
    branch. You might suspect that the 1x1 convolution operation produces an
    output, that is equal to the input. If you think that, then your intuition
    is wrong. Remember that the convolution operation is applied to all feature
    maps in the previous layer. While the width and the height after the 1x1
    convolution remain the same, the number of filters can be changed
    arbitrarily. Below for example we take 4 feature maps as input and return
    just one single feature map.
  </p>

  <Convolution
    imageWidth={6}
    imageHeight={6}
    kernel={1}
    showOutput="true"
    numChannels={4}
    numFilters={1}
  />
  <p>
    This operation allows us to reduce the number of feature maps in order to
    save computational power. This is especially relevant for the 3x3 and 5x5
    filters, as those require a lot of weights, when the number of filter grows.
    That means that in the inception block we reduce the number of filters,
    before we apply the 3x3 and 5x5 filters.
  </p>
  <p>
    You should also bear in mind that in each branch the size of the feature
    maps have to match. If they wouldn't, you would not be able to concatenate
    the branches in the last step. The number of channels after the
    concatenation corresponds to the sum of the channels from the four branches.
  </p>

  <p>
    The overall GoogLeNet architecture combines many layers of Inception and
    Pooling blocks. You can get the exact parameters either by studying the
    original paper.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each header as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each data as row}
        <Row>
          {#each row as cell}
            <DataEntry>
              {#if cell === "Inception"}
                <span class="inline-block bg-red-100 px-3 py-1 rounded-full"
                  >{cell}</span
                >
              {:else if cell === "Max Pooling"}
                <span class="inline-block bg-slate-200 px-3 py-1 rounded-full"
                  >{cell}</span
                >
              {:else if cell === "Basic Block"}
                <span class="inline-block bg-yellow-100 px-3 py-1 rounded-full"
                  >{cell}</span
                >
              {:else}
                {cell}
              {/if}
            </DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    Aside from the inception blocks, there are more details, that we have not
    seen so far. With AlexNet for example we used several fully connected layers
    in the classification block. We did that to slowly move from the flattened
    vector to the number of neurons that are used as input into the softmax
    layer. In the GoogLeNet architecture the last pooling layer removes the
    width and length and we use a single fully connected layer before the
    sigmoid/softmax layer. Such a procedure is quite common nowadays. Fully
    connected layers require many parameters and the approach above avoids
    unnecessary calculations (see Lin et al. <InternalLink
      type={"reference"}
      id={2}
    /> for more info).
  </p>
  <p>
    Be aware that the architecture we have discussed above is often called
    InceptionV1. The field of deep learning improved very fast, which resulted
    in the improved InceptionV2 and InceptionV3<InternalLink
      type={"reference"}
      id={3}
    /> architectues. Below we will implement the original GoogLeNet architecture.
  </p>
  <PythonCode
    code={`import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}
  />
  <PythonCode
    code={`train_transform = T.Compose([T.Resize((50, 50)), 
                             T.ToTensor(),
                             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])`}
  />
  <PythonCode
    code={`train_val_dataset = CIFAR10(root='../datasets', download=True, train=True, transform=train_transform)`}
  />
  <PythonCode
    code={`# split dataset into train and validate
indices = list(range(len(train_val_dataset)))
train_idxs, val_idxs = train_test_split(
    indices, test_size=0.1, stratify=train_val_dataset.targets
)

train_dataset = Subset(train_val_dataset, train_idxs)
val_dataset = Subset(train_val_dataset, val_idxs)`}
  />
  <PythonCode
    code={`batch_size=128
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)`}
  />
  <p>
    The basic block module is going to be used extensively for the inception
    module below.
  </p>
  <PythonCode
    code={`class BasicBlock(nn.Module):
    def __init__(self,
               in_channels,
               out_channels,
               **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      bias=False,
                      **kwargs),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())
  
    def forward(self, x):
        return self.block(x)`}
  />
  <p>
    The four branches of the inception module run separately first, but are then
    concatenated on the channel dimension (<code>dim=1</code>).
  </p>
  <PythonCode
    code={`class InceptionBlock(nn.Module):

    def __init__(self, 
                 in_channels, 
                 conv1x1_channels,
                 conv3x3_input_channels,
                 conv3x3_channels,
                 conv5x5_input_channels,
                 conv5x5_channels,
                 projection_channels):
        super().__init__()
        
        self.branch_1 = BasicBlock(in_channels=in_channels, 
                                  out_channels=conv1x1_channels,
                                  kernel_size=1)
        
        self.branch_2 = nn.Sequential(
            BasicBlock(in_channels=in_channels,
                        out_channels=conv3x3_input_channels,
                        kernel_size=1),
            BasicBlock(in_channels=conv3x3_input_channels,
                        out_channels=conv3x3_channels,
                        kernel_size=3,
                        padding=1))
        
        self.branch_3 = nn.Sequential(
            BasicBlock(in_channels=in_channels, 
                       out_channels=conv5x5_input_channels,
                       kernel_size=1),
            BasicBlock(in_channels=conv5x5_input_channels,
                       out_channels=conv5x5_channels,
                       kernel_size=5,
                       padding=2))
        
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicBlock(in_channels, projection_channels, kernel_size=1),
        )

        
    def forward(self, x):
        return torch.cat([self.branch_1(x), 
                          self.branch_2(x), 
                          self.branch_3(x), 
                          self.branch_4(x)], dim=1)`}
  />
  <p>
    Finally we implement the GoogLeNet module. The parameters used below were
    taken from the original paper. In the forward pass we comment out one of the
    max pooling layers. We do that, because our images are significantly smaller
    than the ImageNet images and we would run into errors if we included the
    additional pooling layer.
  </p>
  <PythonCode
    code={`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = BasicBlock(in_channels=3, 
                                out_channels=64, 
                                kernel_size=7, 
                                stride=2,
                                padding=3)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, 
                                       stride=2, 
                                       ceil_mode=True)

        self.conv_2 = BasicBlock(in_channels=64, 
                                out_channels=64,
                                kernel_size=1)

        self.conv_3 = BasicBlock(in_channels=64, 
                                out_channels=192, 
                                kernel_size=3, 
                                stride=1,
                                padding=1)
    
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_3a = InceptionBlock(
            in_channels=192, 
            conv1x1_channels=64,
            conv3x3_input_channels=96,
            conv3x3_channels=128,
            conv5x5_input_channels=16,
            conv5x5_channels=32,
            projection_channels=32)

        self.inception_3b = InceptionBlock(
            in_channels=256, 
            conv1x1_channels=128,
            conv3x3_input_channels=128,
            conv3x3_channels=192,
            conv5x5_input_channels=32,
            conv5x5_channels=96,
            projection_channels=64)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, 10)


    def forward(self, x):
        x = self.conv_1(x)
        #x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.max_pool_2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_3(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x`}
  />
  <PythonCode
    code={`def track_performance(dataloader, model, criterion):
    # switch to evaluation mode
    model.eval()
    num_samples = 0
    num_correct = 0
    loss_sum = 0

    # no need to calculate gradients
    with torch.inference_mode():
        for _, (features, labels) in enumerate(dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)

                predictions = logits.max(dim=1)[1]
                num_correct += (predictions == labels).sum().item()

                loss = criterion(logits, labels)
                loss_sum += loss.cpu().item()
                num_samples += len(features)

    # we return the average loss and the accuracy
    return loss_sum / num_samples, num_correct / num_samples`}
  />
  <PythonCode
    code={`def train(
    num_epochs,
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    scheduler=None,
):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        start_time = time.time()
        for _, (features, labels) in enumerate(train_dataloader):
            model.train()
            features = features.to(device)
            labels = labels.to(device)

            # Empty the gradients
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Forward Pass
                logits = model(features)
                # Calculate Loss
                loss = criterion(logits, labels)

            # Backward Pass
            scaler.scale(loss).backward()

            # Gradient Descent
            scaler.step(optimizer)
            scaler.update()

        val_loss, val_acc = track_performance(val_dataloader, model, criterion)
        end_time = time.time()

        s = (
            f"Epoch: {epoch+1:>2}/{num_epochs} | "
            f"Epoch Duration: {end_time - start_time:.3f} sec | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Acc: {val_acc:.3f} |"
        )
        print(s)

        if scheduler:
            scheduler.step(val_loss)`}
  />
  <PythonCode
    code={`model = Model()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}
  />
  <p>
    The GoogLeNet model produces an accurancy of close to <Highlight
      >87%</Highlight
    >. We do not beat our VGG implementation, but our runtime is significantly
    reduced.
  </p>
  <PythonCode
    code={`train(
    num_epochs=30,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
)`}
  />
  <PythonCode
    isOutput={true}
    code={`Epoch:  1/30 | Epoch Duration: 21.662 sec | Val Loss: 1.02850 | Val Acc: 0.637 |
Epoch:  2/30 | Epoch Duration: 20.848 sec | Val Loss: 0.86810 | Val Acc: 0.713 |
Epoch:  3/30 | Epoch Duration: 20.950 sec | Val Loss: 0.75014 | Val Acc: 0.744 |
Epoch:  4/30 | Epoch Duration: 21.108 sec | Val Loss: 0.62963 | Val Acc: 0.785 |
Epoch:  5/30 | Epoch Duration: 21.120 sec | Val Loss: 0.62424 | Val Acc: 0.793 |
Epoch:  6/30 | Epoch Duration: 20.876 sec | Val Loss: 0.59486 | Val Acc: 0.814 |
Epoch:  7/30 | Epoch Duration: 20.860 sec | Val Loss: 0.59696 | Val Acc: 0.811 |
Epoch:  8/30 | Epoch Duration: 20.894 sec | Val Loss: 0.60809 | Val Acc: 0.818 |
Epoch:  9/30 | Epoch Duration: 21.068 sec | Val Loss: 0.87457 | Val Acc: 0.769 |
Epoch 00009: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 10/30 | Epoch Duration: 20.961 sec | Val Loss: 0.45824 | Val Acc: 0.868 |
Epoch: 11/30 | Epoch Duration: 20.983 sec | Val Loss: 0.49140 | Val Acc: 0.867 |
Epoch: 12/30 | Epoch Duration: 21.150 sec | Val Loss: 0.51830 | Val Acc: 0.868 |
Epoch: 13/30 | Epoch Duration: 20.836 sec | Val Loss: 0.56201 | Val Acc: 0.869 |
Epoch 00013: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 14/30 | Epoch Duration: 20.986 sec | Val Loss: 0.56474 | Val Acc: 0.868 |
Epoch: 15/30 | Epoch Duration: 20.946 sec | Val Loss: 0.55584 | Val Acc: 0.869 |
Epoch: 16/30 | Epoch Duration: 20.966 sec | Val Loss: 0.56621 | Val Acc: 0.870 |
Epoch 00016: reducing learning rate of group 0 to 1.0000e-06.
Epoch: 17/30 | Epoch Duration: 21.122 sec | Val Loss: 0.57022 | Val Acc: 0.868 |
Epoch: 18/30 | Epoch Duration: 20.916 sec | Val Loss: 0.57037 | Val Acc: 0.870 |
Epoch: 19/30 | Epoch Duration: 21.257 sec | Val Loss: 0.57713 | Val Acc: 0.867 |
Epoch 00019: reducing learning rate of group 0 to 1.0000e-07.
Epoch: 20/30 | Epoch Duration: 20.884 sec | Val Loss: 0.56507 | Val Acc: 0.869 |
Epoch: 21/30 | Epoch Duration: 21.388 sec | Val Loss: 0.56922 | Val Acc: 0.869 |
Epoch: 22/30 | Epoch Duration: 21.201 sec | Val Loss: 0.56943 | Val Acc: 0.869 |
Epoch 00022: reducing learning rate of group 0 to 1.0000e-08.
Epoch: 23/30 | Epoch Duration: 20.873 sec | Val Loss: 0.56877 | Val Acc: 0.869 |
Epoch: 24/30 | Epoch Duration: 21.181 sec | Val Loss: 0.56954 | Val Acc: 0.867 |
Epoch: 25/30 | Epoch Duration: 20.905 sec | Val Loss: 0.56815 | Val Acc: 0.871 |
Epoch: 26/30 | Epoch Duration: 20.958 sec | Val Loss: 0.56737 | Val Acc: 0.869 |
Epoch: 27/30 | Epoch Duration: 21.091 sec | Val Loss: 0.56780 | Val Acc: 0.868 |
Epoch: 28/30 | Epoch Duration: 21.017 sec | Val Loss: 0.56894 | Val Acc: 0.869 |
Epoch: 29/30 | Epoch Duration: 21.117 sec | Val Loss: 0.56901 | Val Acc: 0.871 |
Epoch: 30/30 | Epoch Duration: 20.952 sec | Val Loss: 0.56453 | Val Acc: 0.869 |`}
  />
</Container>

<Footer {references} />
