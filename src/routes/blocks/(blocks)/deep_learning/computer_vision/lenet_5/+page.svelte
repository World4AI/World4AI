<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";
  import Highlight from "$lib/Highlight.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  import cifar10 from "./cifar_10.png";

  let header = [
    "Layer",
    "Input Size",
    "Kernel Size",
    "Stride",
    "Padding",
    "Feature Maps",
    "Output Size",
  ];
  let data = [
    ["Conv", "28x28x1", "5x5", "1", "2", "6", "28x28x6"],
    ["Tanh", "-", "-", "-", "-", "-", "-"],
    ["Avg. Pooling", "28x28x6", "2x2", "2", "0", "-", "14x14x6"],
    ["Conv", "14x14x6", "5x5", "1", "0", 16, "10x10x16"],
    ["Tanh", "-", "-", "-", "-", "-", "-"],
    ["Avg. Pooling", "10x10x16", "2x2", "2", "0", "-", "5x5x16"],
    ["Conv", "5x5x6", "5x5", "1", "0", "120", "1x1x120"],
    ["Tanh", "-", "-", "-", "-", "-", "-"],
    ["FC", "120", "-", "-", "-", "-", "84"],
    ["Tanh", "-", "-", "-", "-", "-", "-"],
    ["FC", "84", "-", "-", "-", "-", "10"],
    ["Softmax", "10", "-", "-", "-", "-", "10"],
  ];

  let references = [
    {
      author: "Y. Lecun, L. Bottou, Y. Bengio and P. Haffner",
      title: "Gradient-based learning applied to document recognition",
      journal: "Proceedings of the IEEE",
      year: "1998",
      pages: "2278-2324",
      volume: "86",
      issue: "11",
    },
  ];
</script>

<svelte:head>
  <title>LeNet-5 - World4AI</title>
  <meta
    name="description"
    content="LeNet-5 is one of the oldest and at the same time one of most well known convolutional neural networks architectures. This architecture was developed by Yann LeCun in 1998 and is still used even to these days to learn the basics of convolutional neural networks."
  />
</svelte:head>

<h1>LeNet-5</h1>
<div class="separator" />
<Container>
  <p>
    LeNet-5<InternalLink type={"reference"} id={1} /> is simultaneouly one of the
    oldest and the most well known convolutional neural network architecture. The
    name LeNet-5 is a reference to the inventor of the network, Yann LeCun, considered
    to be one of the grandfathers of deep learning. The network was designed for
    image recognition and was extensively used with the MNIST dataset. We can not
    expect this architecture to produce state of the art results, but this is a good
    exercise and a great starting point in our study of cnn architectures.
  </p>
  <p>
    The table below depicts the architecure of the LeCun-5 network. First we
    apply 3 convolutional layers and 2 average pooling layers. After the third
    convolutional layer we flatten the feature maps and use two fully connected
    layers. We use the tanh activation function for all convolutional and fully
    connected layers, only the last fully connected layer uses the softmax
    activation function.
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
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    Below we implement LeNet-5 in PyTorch. There are no new pieces in the code
    below, all the piece swere already covered in previous sections.
  </p>
  <PythonCode
    code={`from sklearn.model_selection import train_test_split
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.mnist import MNIST
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}
  />
  <PythonCode
    code={`# -----------------------------------
# DATASETS and DATALOADERS
# -----------------------------------

# get MNIST data
train_val_dataset = MNIST(
    root="../datasets", download=True, train=True, transform=T.ToTensor()
)
test_dataset = MNIST(
    root="../datasets", download=False, train=False, transform=T.ToTensor()
)

# split dataset into train and validate
indices = list(range(len(train_val_dataset)))
train_idxs, val_idxs = train_test_split(
    indices, test_size=0.1, stratify=train_val_dataset.targets.numpy()
)

train_dataset = Subset(train_val_dataset, train_idxs)
val_dataset = Subset(train_val_dataset, val_idxs)

batch_size = 32
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
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)`}
  />
  <p>The model is an exact implementation of the table above.</p>
  <PythonCode
    code={`# -----------------------------------
# LeNet-5 Model
# -----------------------------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding=0),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits`}
  />
  <PythonCode
    code={`# -----------------------------------
# CALCULATE PERFORMANCE
# -----------------------------------
def track_performance(dataloader, model, criterion):
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
    code={`# -----------------------------------
# TRAIN
# -----------------------------------

def train(
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
  <p>
    We use the vanilla gradient descent optimizer, as was done in the original
    paper.
  </p>
  <PythonCode
    code={`model = Model()
optimizer = optim.SGD(params=model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}
  />
  <p>
    Using a LeNet-5 ConvNet allows us to achieve the best performance so far. We
    are close to 99% accuracy on the validation dataset. We could theoretically
    squeeze out a little more performance by for example utilizing data
    augmentation, but this is good enough for MNIST. We consider this task as
    solved and will focus on harder datasets in the next sections to demonstrate
    the usefulnes of more modern convolutional architectures.
  </p>
  <PythonCode
    code={`train(
    num_epochs=10,
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
    code={`Epoch:  1/10 | Epoch Duration: 8.358 sec | Val Loss: 0.07504 | Val Acc: 0.978 |
Epoch:  2/10 | Epoch Duration: 7.312 sec | Val Loss: 0.05162 | Val Acc: 0.984 |
Epoch:  3/10 | Epoch Duration: 7.049 sec | Val Loss: 0.04810 | Val Acc: 0.984 |
Epoch:  4/10 | Epoch Duration: 7.019 sec | Val Loss: 0.04933 | Val Acc: 0.985 |
Epoch:  5/10 | Epoch Duration: 7.180 sec | Val Loss: 0.04957 | Val Acc: 0.987 |
Epoch:  6/10 | Epoch Duration: 7.215 sec | Val Loss: 0.04996 | Val Acc: 0.988 |
Epoch 00006: reducing learning rate of group 0 to 1.0000e-03.
Epoch:  7/10 | Epoch Duration: 7.212 sec | Val Loss: 0.03751 | Val Acc: 0.990 |
Epoch:  8/10 | Epoch Duration: 7.243 sec | Val Loss: 0.03633 | Val Acc: 0.989 |
Epoch:  9/10 | Epoch Duration: 7.464 sec | Val Loss: 0.03620 | Val Acc: 0.989 |
Epoch: 10/10 | Epoch Duration: 7.253 sec | Val Loss: 0.03639 | Val Acc: 0.989 |`}
  />
  <p>
    In the next couple of sections we are going to focus on the so called
    CIFAR-10 dataset. Torchvision provides the dataset out of the box, so let's
    get the data and prepare the dataloaders.
  </p>
  <PythonCode code={`from torchvision.datasets.cifar import CIFAR10`} />
  <PythonCode
    code={`train_val_dataset = CIFAR10(root='../datasets', download=True, train=True, transform=T.ToTensor())
test_dataset = CIFAR10(root='../datasets', download=False, train=False, transform=T.ToTensor())`}
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
    code={`batch_size = 32
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
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)`}
  />
  <p>
    This dataset consists of 45,000 training images, 5,000 validation images and
    10,000 testing images. The images are of size 32x32 pixels and are colored,
    so unlike MNIST we are dealing with a 3-channel input.
  </p>
  <p>
    Once again we are dealing with a classification problem with 10 distinct
    labels.
  </p>
  <PythonCode
    code={`classes = ('plane', 'car', 'bird', 'cat','deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')`}
  />
  <p>
    While the image size is fairly similar to MNIST, the task is significantly
    more complex than MNIST. Look at the images below. Unlike MNIST, CIFAR-10
    consits of real-life images. The dataset is much more diverse. Often you see
    the objects from different angles or distances. The objects often have
    different colors. Moreover the images contain different types of
    backgrounds. Getting a good accuracy is going to be a fairly challenging
    task.
  </p>
  <PythonCode
    code={`fig = plt.figure(figsize=(6, 8))
columns = 4
rows = 5

for i in range(1, columns*rows +1):
    img, cls = train_val_dataset[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.title(classes[cls])
    plt.axis('off')
plt.show()`}
  />
  <img class="mx-auto" src={cifar10} alt="A collection of cifar-10 images" />
  <p>
    Let's use the LeCun-5 architecture, in order to create a simple baseline,
    that we should try to beat with more modern architectures in the next
    sections.
  </p>
  <p>
    Our model is very similar, but not identical. First, the input channels for
    the very first convolutional layer were increased in order to account for
    the colored images. Second, we introduced the <code>AdaptiveAvgPool2d</code>
    layer with the output size of (1,1). This layer applies an average pooling, such
    that the width and height of the image are equal to a given size, in our case
    we reduce the image to just 1x1 pixel. We do that, because the parameters we
    have chosen below correspond to an 28x28 MNIST image and when we input a 32x32
    image, we are left with more than 120 parameters. This layer is often use to
    make one single architecture compatible with different image sizes.
  </p>
  <PythonCode
    code={`# -----------------------------------
# LeNet-5 Model
# -----------------------------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # inputut channels equals to 3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding=0),
            nn.Tanh(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x`}
  />
  <p>
    Let's pretend for a second, that we did'n include the <code
      >AdaptiveAvgPool2d</code
    > module. In that case the feature extractor produces an image of size 2x2.
  </p>
  <PythonCode
    code={`with torch.inference_mode():
    x = torch.randn(1, 3, 32, 32).to(device)
    x = model.feature_extractor(x)
    print(x.shape)`}
  />
  <PythonCode code={`torch.Size([1, 120, 2, 2])`} isOutput={true} />
  <p>
    After flattening the image, we would end up with 120x2x2 features, while the
    linear layer expects exactly 120. The <code>avgpool</code> on the other hand
    always reduces the image to a size of 1x1, no matter if the input is 2x2, 3x3
    or of any other dimension.
  </p>
  <p>
    As usual, we create our model, optimizer, scheduler and criterion and train
    the model.
  </p>
  <PythonCode
    code={`model = Model()
optimizer = optim.SGD(params=model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}
  />
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
    code={`Epoch:  1/30 | Epoch Duration: 4.378 sec | Val Loss: 1.55044 | Val Acc: 0.432 |
Epoch:  2/30 | Epoch Duration: 4.689 sec | Val Loss: 1.39773 | Val Acc: 0.495 |
Epoch:  3/30 | Epoch Duration: 4.552 sec | Val Loss: 1.31610 | Val Acc: 0.529 |
Epoch:  4/30 | Epoch Duration: 4.434 sec | Val Loss: 1.35375 | Val Acc: 0.525 |
Epoch:  5/30 | Epoch Duration: 4.521 sec | Val Loss: 1.27864 | Val Acc: 0.546 |
Epoch:  6/30 | Epoch Duration: 4.576 sec | Val Loss: 1.26108 | Val Acc: 0.554 |
Epoch:  7/30 | Epoch Duration: 4.523 sec | Val Loss: 1.35097 | Val Acc: 0.521 |
Epoch:  8/30 | Epoch Duration: 4.492 sec | Val Loss: 1.25061 | Val Acc: 0.566 |
Epoch:  9/30 | Epoch Duration: 4.675 sec | Val Loss: 1.27933 | Val Acc: 0.549 |
Epoch: 10/30 | Epoch Duration: 4.565 sec | Val Loss: 1.25053 | Val Acc: 0.557 |
Epoch: 11/30 | Epoch Duration: 4.760 sec | Val Loss: 1.25920 | Val Acc: 0.571 |
Epoch: 12/30 | Epoch Duration: 4.596 sec | Val Loss: 1.32838 | Val Acc: 0.540 |
Epoch: 13/30 | Epoch Duration: 4.581 sec | Val Loss: 1.31666 | Val Acc: 0.546 |
Epoch: 14/30 | Epoch Duration: 4.478 sec | Val Loss: 1.27574 | Val Acc: 0.564 |
Epoch 00014: reducing learning rate of group 0 to 1.0000e-03.
Epoch: 15/30 | Epoch Duration: 4.667 sec | Val Loss: 1.17661 | Val Acc: 0.594 |
Epoch: 16/30 | Epoch Duration: 4.651 sec | Val Loss: 1.18592 | Val Acc: 0.599 |
Epoch: 17/30 | Epoch Duration: 4.693 sec | Val Loss: 1.20123 | Val Acc: 0.598 |
Epoch: 18/30 | Epoch Duration: 4.533 sec | Val Loss: 1.21307 | Val Acc: 0.595 |
Epoch: 19/30 | Epoch Duration: 4.688 sec | Val Loss: 1.23074 | Val Acc: 0.597 |
Epoch: 20/30 | Epoch Duration: 4.423 sec | Val Loss: 1.23757 | Val Acc: 0.596 |
Epoch: 21/30 | Epoch Duration: 4.837 sec | Val Loss: 1.26319 | Val Acc: 0.593 |
Epoch 00021: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 22/30 | Epoch Duration: 4.638 sec | Val Loss: 1.25713 | Val Acc: 0.595 |
Epoch: 23/30 | Epoch Duration: 4.472 sec | Val Loss: 1.25976 | Val Acc: 0.593 |
Epoch: 24/30 | Epoch Duration: 4.450 sec | Val Loss: 1.26443 | Val Acc: 0.596 |
Epoch: 25/30 | Epoch Duration: 5.288 sec | Val Loss: 1.26758 | Val Acc: 0.597 |
Epoch: 26/30 | Epoch Duration: 4.750 sec | Val Loss: 1.27124 | Val Acc: 0.596 |
Epoch: 27/30 | Epoch Duration: 4.771 sec | Val Loss: 1.27477 | Val Acc: 0.596 |
Epoch 00027: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 28/30 | Epoch Duration: 4.605 sec | Val Loss: 1.27532 | Val Acc: 0.596 |
Epoch: 29/30 | Epoch Duration: 5.403 sec | Val Loss: 1.27567 | Val Acc: 0.596 |
Epoch: 30/30 | Epoch Duration: 4.882 sec | Val Loss: 1.27617 | Val Acc: 0.596 |`}
  />
  <p>
    After 30 epochs we reach an <Highlight>accuracy of roughly 60%</Highlight> This
    is the number we have to beat.
  </p>
</Container>

<Footer {references} />
