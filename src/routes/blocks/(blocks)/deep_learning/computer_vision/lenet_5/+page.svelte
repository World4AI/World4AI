<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

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
</Container>

<Footer {references} />
