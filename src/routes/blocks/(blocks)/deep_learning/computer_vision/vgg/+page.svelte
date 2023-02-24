<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // table library
  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  //diagram
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  // basic block
  let gap = 1;
  let basicWidth = 90;
  let basicHeight = 100;
  let basicBoxWidth = 80;
  let basicBoxHeight = 20;
  let basicMaxWidth = "200px";

  let header = ["Type", "Input Size", "Output Size"];
  let data = [
    ["VGG Module", "224x224x3", "224x224x64"],
    ["VGG Module", "224x224x64", "224x224x64"],
    ["Max Pooling", "224x224x64", "112x112x64"],
    ["VGG Module", "112x112x64", "112x112x128"],
    ["VGG Module", "112x112x128", "112x112x128"],
    ["Max Pooling", "112x112x128", "56x56x128"],
    ["VGG Module", "56x56x128", "56x56x256"],
    ["VGG Module", "56x56x256", "56x56x256"],
    ["VGG Module", "56x56x256", "56x56x256"],
    ["Max Pooling", "56x56x256", "28x28x256"],
    ["VGG Module", "28x28x256", "28x28x512"],
    ["VGG Module", "28x28x512", "28x28x512"],
    ["VGG Module", "28x28x512", "28x28x512"],
    ["Max Pooling", "28x28x512", "14x14x512"],
    ["VGG Module", "14x14x512", "14x14x512"],
    ["VGG Module", "14x14x512", "14x14x512"],
    ["VGG Module", "14x14x512", "14x14x512"],
    ["Max Pooling", "14x14x512", "7x7x512"],
    ["Dropout", "-", "-"],
    ["Fully Connected", "25088", "4096"],
    ["ReLU", "-", "-"],
    ["Dropout", "-", "-"],
    ["Fully Connected", "4096", "4096"],
    ["ReLU", "-", "-"],
    ["Fully Connected", "4096", "1000"],
    ["Softmax", "-", "-"],
  ];

  let references = [
    {
      author: "Simonyan, K., & Zisserman, A.",
      title:
        "Very deep convolutional networks for large-scale image recognition",
      journal: "",
      year: "2014",
      pages: "",
      volume: "",
      issue: "",
    },
  ];
</script>

<svelte:head>
  <title>VGG - World4AI</title>
  <meta
    name="description"
    content="VGG is at heart a very simple convolutional neural network architecture. It stacks layers of convolutions followed by max pooling. But compared to AlexNet or LeNet-5 this architecture showed that deeper and deeper networks might be necessary to achieve truly impressive results."
  />
</svelte:head>

<h1>VGG</h1>
<div class="separator" />
<Container>
  <p>
    The <Highlight>VGG</Highlight><InternalLink type={"reference"} id={1} /> ConvNet
    architecture was developed by the Visual Geometry Group, a computer vision research
    lab at Oxford university. The neural network is similar in spirit to LeNet-5
    and AlexNet, but VGG is a much deeper neural network. Unlike AlexNet, VGG does
    not apply any large filters, but uses only small patches of 3x3. The authors
    attributed this design choice to the success of their neural network. VGG got
    second place for object classification and first place for object detection in
    the 2014 ImageNet challenge.
  </p>
  <p>
    The VGG paper discussed networks of varying depth, from 11 layers to 19
    layers. We are going to discuss the 16 layer architecture, the so called
    VGG16 (architecture D in the paper).
  </p>

  <p>
    As with many other deep learning architectures, VGG reuses the same module
    over and over again. The VGG module uses a convolutional layer with the
    kernel size of 3x3, stride of size 1 and padding of size 1, followed by
    batch normalization and the ReLU activation function. Be aware, that the
    BatchNorm2d layer was not used in the original VGG paper, but if you omit
    the normalization step, the network might suffer from vanishing gradients.
  </p>
  <p />
  <SvgContainer maxWidth={basicMaxWidth}>
    <svg viewBox="0 0 {basicWidth} {basicHeight}">
      <Block
        x={basicWidth / 2}
        y={basicHeight - basicBoxHeight / 2 - gap}
        width={basicBoxWidth}
        height={basicBoxHeight}
        text="Conv2d: 3x3, S:1, P:1"
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
    After a couple of such modules, we apply a max pooling layer with a kernel
    of 2 and a stride of 2.
  </p>
  <p>The full VGG16 implementation looks as follows.</p>
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
              {#if cell === "VGG Module"}
                <span class="inline-block bg-red-100 px-3 py-1 rounded-full"
                  >{cell}</span
                >
              {:else if cell === "Max Pooling"}
                <span class="inline-block bg-slate-200 px-3 py-1 rounded-full"
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
  <p>Below we implement VGG16 to classify the images in the CIFAR-10 datset.</p>
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
    code={`# In the paper a batch size of 256 was used
batch_size=32
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
    We create a <code>VGG_Block</code> module, that we can reuse many times.
  </p>
  <PythonCode
    code={`class VGG_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(num_features=out_channels),
          nn.ReLU(inplace=True)
        )
  
    def forward(self, x):
        return self.layer(x)`}
  />
  <p>
    VGG has a lot of repeatable blocks. It is common practice to store the
    configuration in a list and to construct the model from the config. The
    numbers represent the number of output filters in a convolutional layer. 'M'
    on the other hand indicates a maxpool layer.
  </p>
  <PythonCode
    code={`cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]`}
  />
  <p>
    Out model implementation is very close to the table above, but we have to
    account for the fact that our images are smaller, so we reduce the input in
    the first linear layer from 7x7x512 to 1x1x512.
  </p>
  <PythonCode
    code={`class Model(nn.Module):

    def __init__(self, cfg, num_classes=1):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = self._make_feature_extractor()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )
        
    def _make_feature_extractor(self):
        layers = []
        in_channels = 3
        for element in self.cfg:
            if element == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [VGG_Block(in_channels, element)]
                in_channels = element
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
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
    code={`model = Model(cfg)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}
  />
  <p>
    When we train VGG16 on the CIFAR-10 dataset, we reach an accuracy of roughly <Highlight
      >86%</Highlight
    >, thereby beating the LeCun-5 and the AlexNet implementation.
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
    code={`Epoch:  1/30 | Epoch Duration: 44.329 sec | Val Loss: 1.59811 | Val Acc: 0.369 |
Epoch:  2/30 | Epoch Duration: 43.818 sec | Val Loss: 1.28220 | Val Acc: 0.528 |
Epoch:  3/30 | Epoch Duration: 43.788 sec | Val Loss: 1.07582 | Val Acc: 0.616 |
Epoch:  4/30 | Epoch Duration: 43.785 sec | Val Loss: 0.88148 | Val Acc: 0.684 |
Epoch:  5/30 | Epoch Duration: 43.857 sec | Val Loss: 0.83410 | Val Acc: 0.713 |
Epoch:  6/30 | Epoch Duration: 44.058 sec | Val Loss: 0.73667 | Val Acc: 0.760 |
Epoch:  7/30 | Epoch Duration: 43.944 sec | Val Loss: 0.68140 | Val Acc: 0.771 |
Epoch:  8/30 | Epoch Duration: 43.986 sec | Val Loss: 0.66568 | Val Acc: 0.775 |
Epoch:  9/30 | Epoch Duration: 44.200 sec | Val Loss: 0.60432 | Val Acc: 0.800 |
Epoch: 10/30 | Epoch Duration: 44.085 sec | Val Loss: 0.56420 | Val Acc: 0.811 |
Epoch: 11/30 | Epoch Duration: 43.931 sec | Val Loss: 0.55904 | Val Acc: 0.816 |
Epoch: 12/30 | Epoch Duration: 43.955 sec | Val Loss: 0.55706 | Val Acc: 0.831 |
Epoch: 13/30 | Epoch Duration: 43.865 sec | Val Loss: 0.53854 | Val Acc: 0.833 |
Epoch: 14/30 | Epoch Duration: 43.942 sec | Val Loss: 0.54444 | Val Acc: 0.833 |
Epoch: 15/30 | Epoch Duration: 44.052 sec | Val Loss: 0.56500 | Val Acc: 0.829 |
Epoch: 16/30 | Epoch Duration: 44.420 sec | Val Loss: 0.59908 | Val Acc: 0.827 |
Epoch 00016: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 17/30 | Epoch Duration: 44.404 sec | Val Loss: 0.55250 | Val Acc: 0.861 |
Epoch: 18/30 | Epoch Duration: 44.398 sec | Val Loss: 0.59180 | Val Acc: 0.861 |
Epoch: 19/30 | Epoch Duration: 44.369 sec | Val Loss: 0.63396 | Val Acc: 0.859 |
Epoch 00019: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 20/30 | Epoch Duration: 44.336 sec | Val Loss: 0.64396 | Val Acc: 0.861 |
Epoch: 21/30 | Epoch Duration: 44.284 sec | Val Loss: 0.66723 | Val Acc: 0.864 |
Epoch: 22/30 | Epoch Duration: 44.499 sec | Val Loss: 0.67943 | Val Acc: 0.861 |
Epoch 00022: reducing learning rate of group 0 to 1.0000e-06.
Epoch: 23/30 | Epoch Duration: 44.465 sec | Val Loss: 0.68677 | Val Acc: 0.863 |
Epoch: 24/30 | Epoch Duration: 44.003 sec | Val Loss: 0.66321 | Val Acc: 0.864 |
Epoch: 25/30 | Epoch Duration: 44.001 sec | Val Loss: 0.66218 | Val Acc: 0.861 |
Epoch 00025: reducing learning rate of group 0 to 1.0000e-07.
Epoch: 26/30 | Epoch Duration: 43.933 sec | Val Loss: 0.67842 | Val Acc: 0.862 |
Epoch: 27/30 | Epoch Duration: 43.856 sec | Val Loss: 0.68216 | Val Acc: 0.862 |
Epoch: 28/30 | Epoch Duration: 43.897 sec | Val Loss: 0.67129 | Val Acc: 0.863 |
Epoch 00028: reducing learning rate of group 0 to 1.0000e-08.
Epoch: 29/30 | Epoch Duration: 43.911 sec | Val Loss: 0.68369 | Val Acc: 0.862 |
Epoch: 30/30 | Epoch Duration: 44.182 sec | Val Loss: 0.67892 | Val Acc: 0.861 |`}
  />
</Container>

<Footer {references} />
