<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  let header = [
    "Type",
    "Input Size",
    "Kernel Size",
    "Stride",
    "Padding",
    "Feature Maps",
    "Output Size",
  ];
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
    ["Dropout", "-", "-", "-", "-", "-", "-"],
    ["Fully Connected", "9219", "-", "-", "-", "-", "4096"],
    ["ReLU", "-", "-", "-", "-", "-", "-"],
    ["Dropout", "-", "-", "-", "-", "-", "-"],
    ["Fully Connected", "4096", "-", "-", "-", "-", "4096"],
    ["ReLU", "-", "-", "-", "-", "-", "-"],
    ["Fully Connected", "4096", "-", "-", "-", "-", "1000"],
    ["Softmax", "-", "-", "-", "-", "-", "-"],
  ];

  let references = [
    {
      author: "Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E",
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      journal: "Advances in Neural Information Processing Systems",
      year: "2012",
      pages: "",
      volume: "25",
      issue: "",
    },
  ];
</script>

<svelte:head>
  <title>AlexNet - World4AI</title>
  <meta
    name="description"
    content="AlexNet is the convolutional network architecture that initiated the latest AI spring. AlexNet showed for the very first time, that the combination of a large amount of data, computational resources and the advances in neural network architectures can produce state of the art results and outperform any other approaches."
  />
</svelte:head>

<h1>AlexNet</h1>
<div class="separator" />
<Container>
  <p>
    <Highlight>AlexNet</Highlight><InternalLink type={"reference"} id={1} /> is a
    ConvNet named after one of its creator, Alex Krizhevsky. Together with Ilya Sutskever
    and Geoffrey Hinton, he won the 2012 ILSVRC (ImageNet Large Scale Visual Recognition
    Challenge).
  </p>
  <p>
    The success of each ImageNet model is determined using top-1 and top-5 error
    rates. Your model has to output the classes with the five highest
    probabilities. For the top-1 error rate the model is considered to be
    successful, if the actual class corresponds to the prediction with the
    highest probability, while for the top-5 error rate the model is considered
    to be successful if the actual class is somewhere in the top-5 predictions.
    In the 2012 challenge AlexNet achieved a top-5 error rate of 15.3%, while
    the second best entry achieved only 26.2%. Prior to that achievement the
    public did not believe that neural networks are capable of such success.
    This became known as the ImageNet moment. From that point on, neural
    networks became mainstream and the current AI spring was started.
  </p>
  <p>
    There were a couple of reasons, that made the success of AlexNet possible.
    Let's go over them.
  </p>
  <p>
    Deep learning models tend to get better, the more data is used for their
    training. AlexNet was trained on a huge amount of data. The ImageNet
    challenge dataset is a subset of the ImageNet database and consists of
    1,281,167 training images, 50,000 validation images and 100,000 test images
    with 1,000 different image categories. While this is relatively modest in
    modern terms, compared to older datasets like MNIST, this was a big step in
    the right direction.
  </p>
  <p>
    Nowadays you can use PyTorch or TensorFlow to write your code in Python and
    you can ignore the low-level GPU implementation, as the deep learning
    framework takes care of that for you. The AlexNet creators did not have that
    luxury. They had to write a custom GPU implementation. While the speedup
    that came with GPU convolutions was huge, they still had to wait 5-6 days
    for the training to finish.
  </p>
  <p>
    The researchers also used relatively modern deep learning techniques for the
    time. For example they used ReLU as activation functions, Dropout to deal
    with overfitting and normalization for some of the convolutional layers.
  </p>
  <p>
    Finally, AlexNet was a much larger model than what was previously thought to
    be possible. When you look at the architecture below, you will notice, that
    for the most part AlexNet does not differ that much from the LeNet-5
    architecture, but it's deeper (more layers) and wider (more channels).
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
    The AlexNet architecture utilized so called local response normalization.
    This normalization step is not used in practice any more and is considered a
    historical artifact. We will not cover this step in detail, because we
    instead utilize the BatchNorm2d layers. The 2d batch normalization differs
    slightly from the 1d version. We calculate one mean and one variance value
    per channel and not per a single value in the channel.
  </p>
  <p>
    The authors preprocessed the data and used image augmentation in order to
    facilitate the training. As the original images in the ImageNet dataset are
    of different size and relatively large, they scaled the images to a 256x256
    pixels and took a random patch of 224x224, that was used as the input into
    the neural network.
  </p>
  <p>
    The max pooling layers in AlexNet are also very unusual. Normally the kernel
    size and the stride are of equivalent size, such that the max value is
    calculated on non overlapping windows. AlexNet on the other hand utilizes
    overlapping max pooling. You will hardly find such an implementation in more
    recent convolutional neural networks.
  </p>
  <p>
    If you want to reimplement the original model from scratch, we recommend you
    study the research paper. We try to stick as much as possible to the
    original model, but from time to time we will deviate, when we deem the
    change as necessary. Let's finally implement AlexNet and train the network
    using the CIFAR-10 dataset.
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
  <p>
    We apply a couple of transforms. Especially we need to resize the image,
    otherwise we would run in errors, as the amount of convolutions and pooling
    layers would basically reduce the image size to 0. An image of 50x50 pixels
    is large enough to run through AlexNet without errors.
  </p>
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
    code={`# The batch size of 128 images is taken from the original paper.
batch_size=128
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
    The model is similar to the one in the paper, but in AlexNet the last
    pooling layer produces a 256x6x6 image, while our image is of size 256x1x1,
    due to small CIFAR-10 images. We account for that and reduce the number of
    input features into the first linear layer.
  </p>
  <PythonCode
    code={`class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            # in AlexNet we would have used 6x6x256
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )
        
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
    code={`model = Model()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}
  />
  <p>
    We train for 30 epochs and produce an accuracy of close to <Highlight
      >75%</Highlight
    >. This is significantly better than the LeNet-5 performance.
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
    code={`Epoch:  1/30 | Epoch Duration: 6.760 sec | Val Loss: 1.61129 | Val Acc: 0.389 |
Epoch:  2/30 | Epoch Duration: 5.634 sec | Val Loss: 1.30560 | Val Acc: 0.522 |
Epoch:  3/30 | Epoch Duration: 5.695 sec | Val Loss: 1.24179 | Val Acc: 0.551 |
Epoch:  4/30 | Epoch Duration: 5.671 sec | Val Loss: 1.18613 | Val Acc: 0.593 |
Epoch:  5/30 | Epoch Duration: 5.686 sec | Val Loss: 1.07944 | Val Acc: 0.624 |
Epoch:  6/30 | Epoch Duration: 5.742 sec | Val Loss: 1.01913 | Val Acc: 0.642 |
Epoch:  7/30 | Epoch Duration: 5.729 sec | Val Loss: 1.00196 | Val Acc: 0.649 |
Epoch:  8/30 | Epoch Duration: 5.728 sec | Val Loss: 0.96261 | Val Acc: 0.682 |
Epoch:  9/30 | Epoch Duration: 5.761 sec | Val Loss: 0.92294 | Val Acc: 0.689 |
Epoch: 10/30 | Epoch Duration: 5.785 sec | Val Loss: 0.95872 | Val Acc: 0.690 |
Epoch: 11/30 | Epoch Duration: 5.744 sec | Val Loss: 0.93677 | Val Acc: 0.692 |
Epoch: 12/30 | Epoch Duration: 5.781 sec | Val Loss: 1.03572 | Val Acc: 0.669 |
Epoch 00012: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 13/30 | Epoch Duration: 5.758 sec | Val Loss: 0.86987 | Val Acc: 0.734 |
Epoch: 14/30 | Epoch Duration: 5.712 sec | Val Loss: 0.87811 | Val Acc: 0.732 |
Epoch: 15/30 | Epoch Duration: 5.748 sec | Val Loss: 0.91566 | Val Acc: 0.731 |
Epoch: 16/30 | Epoch Duration: 5.750 sec | Val Loss: 0.95146 | Val Acc: 0.730 |
Epoch 00016: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 17/30 | Epoch Duration: 5.758 sec | Val Loss: 0.95101 | Val Acc: 0.733 |
Epoch: 18/30 | Epoch Duration: 5.755 sec | Val Loss: 0.95624 | Val Acc: 0.734 |
Epoch: 19/30 | Epoch Duration: 5.753 sec | Val Loss: 0.96301 | Val Acc: 0.732 |
Epoch 00019: reducing learning rate of group 0 to 1.0000e-06.
Epoch: 20/30 | Epoch Duration: 5.735 sec | Val Loss: 0.95980 | Val Acc: 0.735 |
Epoch: 21/30 | Epoch Duration: 5.730 sec | Val Loss: 0.96182 | Val Acc: 0.733 |
Epoch: 22/30 | Epoch Duration: 5.762 sec | Val Loss: 0.96726 | Val Acc: 0.733 |
Epoch 00022: reducing learning rate of group 0 to 1.0000e-07.
Epoch: 23/30 | Epoch Duration: 5.747 sec | Val Loss: 0.96622 | Val Acc: 0.733 |
Epoch: 24/30 | Epoch Duration: 5.780 sec | Val Loss: 0.96463 | Val Acc: 0.734 |
Epoch: 25/30 | Epoch Duration: 5.787 sec | Val Loss: 0.96622 | Val Acc: 0.733 |
Epoch 00025: reducing learning rate of group 0 to 1.0000e-08.
Epoch: 26/30 | Epoch Duration: 5.790 sec | Val Loss: 0.96433 | Val Acc: 0.733 |
Epoch: 27/30 | Epoch Duration: 5.737 sec | Val Loss: 0.96556 | Val Acc: 0.732 |
Epoch: 28/30 | Epoch Duration: 5.785 sec | Val Loss: 0.96416 | Val Acc: 0.733 |
Epoch: 29/30 | Epoch Duration: 5.795 sec | Val Loss: 0.96417 | Val Acc: 0.734 |
Epoch: 30/30 | Epoch Duration: 5.799 sec | Val Loss: 0.96674 | Val Acc: 0.734 |`}
  />
</Container>

<Footer {references} />
