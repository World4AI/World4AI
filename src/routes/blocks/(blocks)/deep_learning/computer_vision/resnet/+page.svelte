<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Convolution from "../../computer_vision/_convolution/Convolution.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  //diagram
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";
  import Plus from "$lib/diagram/Plus.svelte";

  // table library
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  let header = ["Type", "Repeat", "Parameters"];
  let data = [
    ["Convolution 2D", "", "7x7x64"],
    ["BatchNorm2D", "", ""],
    ["ReLU", "", ""],
    ["Max Pooling", "", "Filter: 3x3, Stride: 2"],
    ["ResNet Block", "3", "3x3x64"],
    ["ResNet Block", "4", "3x3x128"],
    ["ResNet Block", "6", "3x3x256"],
    ["ResNet Block", "3", "3x3x512"],
    ["Adaptive Avg. Pooling", "", "512"],
    ["Fully Connected", "", "1000"],
    ["Softmax", "", "1000"],
  ];

  const references = [
    {
      author: "K. He, X. Zhang, S. Ren and J. Sun",
      title: "Deep Residual Learning for Image Recognition",
      journal:
        "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      year: "2016",
      pages: "770-778",
      volume: "",
      issue: "",
    },
  ];

  let gap = 1;

  // basic block
  let basicWidth = 100;
  let basicHeight = 300;
  let basicBoxWidth = 60;
  let basicBoxHeight = 20;
  let basicMaxWidth = "220px";

  const numComponents = 9;
  const vertGap =
    (basicHeight - numComponents - 1 * basicBoxHeight) / (numComponents - 2);
</script>

<svelte:head>
  <title>ResNet - World4AI</title>
  <meta
    name="description"
    content="The ResNet convolutional neural network architecture introduced skip connections. Skip connections allowed to train very deep neural networks and the 152 layer ResNet won the 2015 ImageNet classification competition."
  />
</svelte:head>

<h1>ResNet</h1>
<div class="separator" />
<Container>
  <p>
    We have introduced and discussed <a
      href="/blocks/deep_learning/stability_speedup/skip_connections"
      >skip connections</a
    >
    in a previous chapter. This time around we will talk about <Highlight
      >ResNet</Highlight
    ><InternalLink type={"reference"} id={1} />, the ConvNet which introduced
    skip connections to the world. Several ResNet variants were introduced in
    the original paper. From ResNet18 with just 18 layers all the way to
    ResNet152, with 152 layers. The 152 layer variant won the ILSVRC15
    classification challenge with a 3.57 top-5 error rate. Remember that just
    the year before GoogLeNet achieved 6.67.
  </p>
  <p>
    In this section we will focus on the ResNet34 architecture. But if you are
    interested in implementing the 152 layer architecture, you should be able to
    extend the code below.
  </p>
  <p>
    Similar to the architectures we studied before, ResNet34 is based on many
    basic building blocks, only this time the block is based on skip
    connections.
  </p>
  <SvgContainer maxWidth={basicMaxWidth}>
    <svg viewBox="0 0 {basicWidth} {basicHeight}">
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="Input"
      />
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="Conv2d"
      />
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap * 2}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="BatchNorm2d"
      />
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap * 3}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="ReLU"
      />
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap * 4}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="Conv2d"
      />
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap * 5}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="BatchNorm"
      />
      <Plus
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap * 6}
        radius={5}
        offset={2}
      />
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap - vertGap * 7}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="ReLU"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap + 3,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - vertGap,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap * 2 + 3,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - vertGap * 2,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap * 3 + 3,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - vertGap * 3,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap * 4 + 3,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - vertGap * 4,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap * 5 + 3,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - vertGap * 5,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap * 6,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2,
            y: basicHeight - basicBoxHeight - vertGap * 6,
          },
          {
            x: basicWidth / 2,
            y: basicHeight - vertGap * 7 + 3,
          },
        ]}
        dashed="true"
        moving="true"
      />
      <Arrow
        data={[
          {
            x: basicWidth / 2 - basicBoxWidth / 2,
            y: basicHeight - basicBoxHeight + basicBoxHeight / 2,
          },
          {
            x: basicWidth / 2 - basicBoxWidth / 2 - 15,
            y: basicHeight - basicBoxHeight + basicBoxHeight / 2,
          },
          {
            x: basicWidth / 2 - basicBoxWidth / 2 - 15,
            y: basicHeight - basicBoxHeight + basicBoxHeight / 2 - vertGap * 6,
          },
          {
            x: basicWidth / 2 - 10,
            y: basicHeight - basicBoxHeight + basicBoxHeight / 2 - vertGap * 6,
          },
        ]}
        dashed="true"
        moving="true"
      />
    </svg>
  </SvgContainer>
  <p>
    The block consists of two convolutions. The skip connection goes directly
    from the input to the block (output of the previous layer), past the two
    convolutions and is added to the usual path, before the ReLU is applied to
    the sum. Bear in mind, that this block is slightly different for larger
    ResNet architectures.
  </p>
  <p>
    The number of filters and the image resolution usually stays constant within
    the block. This makes the output size equal to the input size and we do not
    have any trouble adding the input to the output of the second convolution.
    Yet sometimes we reduce the resolution by 2 using a stride of 2 and we
    simultaneously increase the number of filters by two. If we have a 100x100x3
    image, we would end up with a 50x50x6 image. This procedure keeps the number
    of paramters constant, yet we can not apply the addition, because the
    dimensionality of the input and the output differ. In order to deal with the
    problem the input is also processed using a 1x1 kernel with a stride of 2.
  </p>
  <Convolution
    maxWidth={350}
    kernel={1}
    stride={2}
    padding={0}
    imageWidth={6}
    imageHeight={6}
    showOutput={true}
    numChannels={2}
    numFilters={4}
  />
  <p>
    The overall architecture looks as follows. We use the same building blocks
    over and over again. From time to time we halve the resolution and double
    the number of channels.
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
          {#each row as cell, idx}
            <DataEntry>
              {#if cell === "ResNet Block"}
                <span class="inline-block bg-red-100 px-3 py-1 rounded-full"
                  >{cell}</span
                >
              {:else if idx === 1 && cell !== ""}
                <span class="inline-block bg-blue-100 px-3 py-1 rounded-full"
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
    At the end we have a single fully connected layer, which produces 1,000
    logits (the number of ImageNet categories).
  </p>
  <p>
    Let's implement the ResNet34 in PyTorch and train it on the CIFAR-10
    dataset.
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
    Below we implement the skip connection block from the diagram above. If the
    number of input channels and the number of output channels is identical, we
    proceed by adding the calculated residual to the input. If on the other hand
    the number of channels is different, we first use the stride of 2, in order
    to downsample the image by a factor of 2 and we adjust the input through a
    1x1 convolution with stride 2, in order for the addition to work.
  </p>
  <PythonCode
    code={`class BasicBlock(nn.Module):
    def __init__(self,
               in_channels,
               out_channels):
        super().__init__()
    
        first_stride=1
        if out_channels != in_channels:
            first_stride=2

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=first_stride, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(out_channels))

        self.downsampling = None
        if out_channels != in_channels:
            self.downsampling = nn.Sequential(
              nn.Conv2d(in_channels, 
                        out_channels,
                        kernel_size=1,
                        stride=2,
                        bias=False),
              nn.BatchNorm2d(out_channels))
  
    def forward(self, x):
        identity = x
        if self.downsampling:
            identity = self.downsampling(identity)

        x = self.residual(x)
        return torch.relu(x + identity)`}
  />
  <p>
    The configuration list below, contains the number of channels for all of the
    ResNet basic blocks.
  </p>
  <PythonCode
    code={`cfg = [64, 64, 64,
       128, 128, 128, 128,
       256, 256, 256, 256, 256, 256, 
       512, 512, 512]`}
  />
  <p>
    Finally we create a full ResNet34 architecture, using the configuration
    above.
  </p>
  <PythonCode
    code={`class Model(nn.Module):
  
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.blocks = self._create_network()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg[-1], 10))

    def _create_network(self):
        blocks = []
        prev_channels = 64
        for channels in self.cfg:
            blocks += [BasicBlock(prev_channels, channels)]
            prev_channels = channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x`}
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
    code={`model = Model(cfg)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}
  />
  <p>
    Using ResNet34 we achieve an accuracy of roughly <Highlight>83%</Highlight>.
    While we do not beat our previous implementations, do not underestimate skip
    connections. CIFAR-10 is a relatively small dataset, and ResNet34 is a
    relatively small neural network and we therefore can not generalize our
    results. Most modern architectures use skip connections, because this
    technique stood the test of time.
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
    code={`Epoch:  1/30 | Epoch Duration: 12.656 sec | Val Loss: 1.18379 | Val Acc: 0.577 |
Epoch:  2/30 | Epoch Duration: 11.668 sec | Val Loss: 1.02777 | Val Acc: 0.643 |
Epoch:  3/30 | Epoch Duration: 11.689 sec | Val Loss: 0.80600 | Val Acc: 0.726 |
Epoch:  4/30 | Epoch Duration: 11.673 sec | Val Loss: 0.83877 | Val Acc: 0.719 |
Epoch:  5/30 | Epoch Duration: 11.810 sec | Val Loss: 0.70858 | Val Acc: 0.762 |
Epoch:  6/30 | Epoch Duration: 11.836 sec | Val Loss: 0.67763 | Val Acc: 0.779 |
Epoch:  7/30 | Epoch Duration: 11.989 sec | Val Loss: 0.69613 | Val Acc: 0.773 |
Epoch:  8/30 | Epoch Duration: 11.947 sec | Val Loss: 0.65614 | Val Acc: 0.793 |
Epoch:  9/30 | Epoch Duration: 11.867 sec | Val Loss: 0.72713 | Val Acc: 0.784 |
Epoch: 10/30 | Epoch Duration: 11.809 sec | Val Loss: 0.75445 | Val Acc: 0.791 |
Epoch: 11/30 | Epoch Duration: 12.118 sec | Val Loss: 0.80031 | Val Acc: 0.787 |
Epoch 00011: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 12/30 | Epoch Duration: 11.843 sec | Val Loss: 0.65141 | Val Acc: 0.832 |
Epoch: 13/30 | Epoch Duration: 11.893 sec | Val Loss: 0.68573 | Val Acc: 0.828 |
Epoch: 14/30 | Epoch Duration: 11.871 sec | Val Loss: 0.73215 | Val Acc: 0.832 |
Epoch: 15/30 | Epoch Duration: 12.073 sec | Val Loss: 0.77129 | Val Acc: 0.832 |
Epoch 00015: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 16/30 | Epoch Duration: 11.843 sec | Val Loss: 0.77902 | Val Acc: 0.833 |
Epoch: 17/30 | Epoch Duration: 12.498 sec | Val Loss: 0.78212 | Val Acc: 0.831 |
Epoch: 18/30 | Epoch Duration: 11.925 sec | Val Loss: 0.79099 | Val Acc: 0.834 |
Epoch 00018: reducing learning rate of group 0 to 1.0000e-06.
Epoch: 19/30 | Epoch Duration: 11.876 sec | Val Loss: 0.78918 | Val Acc: 0.833 |
Epoch: 20/30 | Epoch Duration: 11.899 sec | Val Loss: 0.79028 | Val Acc: 0.831 |
Epoch: 21/30 | Epoch Duration: 11.871 sec | Val Loss: 0.79865 | Val Acc: 0.832 |
Epoch 00021: reducing learning rate of group 0 to 1.0000e-07.
Epoch: 22/30 | Epoch Duration: 12.021 sec | Val Loss: 0.79126 | Val Acc: 0.831 |
Epoch: 23/30 | Epoch Duration: 11.897 sec | Val Loss: 0.79015 | Val Acc: 0.832 |
Epoch: 24/30 | Epoch Duration: 11.919 sec | Val Loss: 0.78823 | Val Acc: 0.832 |
Epoch 00024: reducing learning rate of group 0 to 1.0000e-08.
Epoch: 25/30 | Epoch Duration: 11.945 sec | Val Loss: 0.78385 | Val Acc: 0.831 |
Epoch: 26/30 | Epoch Duration: 12.040 sec | Val Loss: 0.79242 | Val Acc: 0.831 |
Epoch: 27/30 | Epoch Duration: 11.758 sec | Val Loss: 0.78959 | Val Acc: 0.832 |
Epoch: 28/30 | Epoch Duration: 11.754 sec | Val Loss: 0.79259 | Val Acc: 0.830 |
Epoch: 29/30 | Epoch Duration: 11.838 sec | Val Loss: 0.79554 | Val Acc: 0.831 |
Epoch: 30/30 | Epoch Duration: 11.856 sec | Val Loss: 0.78739 | Val Acc: 0.833 |
`}
  />
</Container>

<Footer {references} />
