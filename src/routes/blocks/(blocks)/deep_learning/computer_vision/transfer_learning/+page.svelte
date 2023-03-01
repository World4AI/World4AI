<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import pets from "./pets.png";
</script>

<svelte:head>
  <title>Transfer Learning - World4AI</title>
  <meta
    name="description"
    content="To train computer vision models on real life tasks requires a lot of computational power and data, whch might not be available. Transfer learning allows us to take a pretrained model and to tune it for our purposes. Transfer learning often workes even if we have a lower end computer and a few samles available."
  />
</svelte:head>

<h1>Transfer Learning</h1>
<div class="separator" />
<Container>
  <p>
    Often our datasets are extremely small and/or we do not have the compute to
    train a large model from scratch.
  </p>
  <Alert type="info">
    You should utilize transfer learning when you do not have the necessary data
    or computational power at your disposal to train large models from scratch
  </Alert>
  <p>
    <Highlight>Transfer learning</Highlight> allows you to take already existing
    pretrained models and to adjust them to your needs. The requirements towards
    computational resources and availability of data sinks dramatically once you
    start to you utilize transfer learning.
  </p>
  <p>
    There are generally two ways to utilize transfer learing: <Highlight
      >feature extraction</Highlight
    > and <Highlight>fine-tuning</Highlight>.
  </p>
  <p>
    When we use the pretrained model as a feature extractor, we load the model,
    freeze all weights and replace the last couple of layers with the layers
    that suit our task. As this procedure only requires to train a few layers,
    it tends to be relatively fast.
  </p>
  <p>
    When we use fine-tuning, we load the weights, replace the last couple of
    layers, but fune-tune all available weights during the training process.
    There is a potential chance to get better results with fine-tuning, but this
    procedure obviously requires more compute.
  </p>
  <p>
    The resoning behind the success of transfer learning is as follows. We have
    mentioned before that the convolutional layers are supposed to learn the
    features of the dataset. It can be argued that if the network has learned to
    recognize edges, colors and higher level features, that those features are
    also useful for other tasks. If the model has learned to classify cats and
    dogs, it should be a relative minor undertaking to adjust the model to
    recognize other animals. On the other hand it is going to be harder to
    fine-tune the same model on a car dataset. The closer the original datset is
    to your data, the more sense it makes to use the pretrained model.
  </p>
  <p>
    For our presentation we have chosen the <a
      href="https://www.robots.ox.ac.uk/~vgg/data/pets/"
      rel="noreferrer"
      target="_blank">Oxford-IIIT Pet Dataset</a
    >. The daset consists of roughly 7400 samples of cats and dogs. There are 37
    categories of cat and dog breeds in the dataset with roughly 200 per
    category. As we will divide the dataset into the training and the validation
    dataset, there will be roughly 100 samples per category for training. All
    things considered, this is a relatively small dataset. We have chosen this
    dataset, because the original ImageNet contains cats and dogs and we expect
    transfer learning to work quite well.
  </p>
  <PythonCode
    code={`import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}
  />
  <PythonCode
    code={`dataset = OxfordIIITPet(root='../datasets', 
                              split='trainval', 
                              target_types='category', 
                              download=True)`}
  />
  <PythonCode
    code={`fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5

for i in range(1, columns*rows +1):
    img, cls = dataset[i*50]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(f'Category {cls}')
    plt.axis('off')
plt.savefig('pets', bbox_inches='tight')
plt.show()`}
  />
  <img src={pets} alt="Different breeds of cats and dogs" />
  <p>
    We will be using the ResNet34 architecture for transfer learning. We can get
    a lot of pretrained computer vision models, including ResNet, from the <code
      >torchvision</code
    >
    library.
  </p>
  <PythonCode
    code={`from torchvision.models import resnet34, ResNet34_Weights`}
  />
  <p>
    When we use transfer learning it is important to utilize the same
    preprocessing steps that were used for the training of the original model.
  </p>
  <PythonCode
    code={`weights = ResNet34_Weights.DEFAULT
preprocess = weights.transforms()`}
  />
  <PythonCode
    code={`train_dataset = OxfordIIITPet(root='../datasets', 
                                  split='trainval', 
                                  target_types='category', 
                                  transform=preprocess, 
                                  download=True)

val_dataset = OxfordIIITPet(root='../datasets', 
                                  split='test', 
                                  target_types='category', 
                                  transform=preprocess, 
                                  download=True)`}
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
  <div class="separator" />
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
  <p>
    We create the ResNet34 model and download the weights that were pretrained
    on the ImageNet dataset.
  </p>
  <PythonCode
    code={`model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)`}
  />
  <p>
    We will utilize the model as a feature extractor, therefore we freeze all
    layer weights.
  </p>
  <PythonCode
    code={`for param in model.parameters():
    param.requires_grad = False`}
  />
  <p>
    We replace the very last layer with a linear layer with 37 outputs. This is
    the only layer that is going to be trained.
  </p>
  <PythonCode code={`model.fc = nn.Linear(in_features=512, out_features=37)`} />
  <PythonCode
    code={`optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
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
    code={`Epoch:  1/30 | Epoch Duration: 11.154 sec | Val Loss: 1.81516 | Val Acc: 0.677 |
Epoch:  2/30 | Epoch Duration: 11.453 sec | Val Loss: 0.90655 | Val Acc: 0.854 |
Epoch:  3/30 | Epoch Duration: 11.348 sec | Val Loss: 0.63867 | Val Acc: 0.870 |
Epoch:  4/30 | Epoch Duration: 11.845 sec | Val Loss: 0.52753 | Val Acc: 0.883 |
Epoch:  5/30 | Epoch Duration: 12.005 sec | Val Loss: 0.46197 | Val Acc: 0.892 |
Epoch:  6/30 | Epoch Duration: 11.932 sec | Val Loss: 0.42866 | Val Acc: 0.894 |
Epoch:  7/30 | Epoch Duration: 12.047 sec | Val Loss: 0.40674 | Val Acc: 0.896 |
Epoch:  8/30 | Epoch Duration: 12.032 sec | Val Loss: 0.38285 | Val Acc: 0.899 |
Epoch:  9/30 | Epoch Duration: 12.055 sec | Val Loss: 0.37018 | Val Acc: 0.900 |
Epoch: 10/30 | Epoch Duration: 11.667 sec | Val Loss: 0.35984 | Val Acc: 0.901 |
Epoch: 11/30 | Epoch Duration: 12.243 sec | Val Loss: 0.34247 | Val Acc: 0.902 |
Epoch: 12/30 | Epoch Duration: 12.104 sec | Val Loss: 0.34527 | Val Acc: 0.900 |
Epoch: 13/30 | Epoch Duration: 12.275 sec | Val Loss: 0.34026 | Val Acc: 0.901 |
Epoch: 14/30 | Epoch Duration: 11.949 sec | Val Loss: 0.33695 | Val Acc: 0.896 |
Epoch: 15/30 | Epoch Duration: 12.117 sec | Val Loss: 0.32628 | Val Acc: 0.904 |
Epoch: 16/30 | Epoch Duration: 11.852 sec | Val Loss: 0.32397 | Val Acc: 0.901 |
Epoch: 17/30 | Epoch Duration: 12.116 sec | Val Loss: 0.32091 | Val Acc: 0.904 |
Epoch: 18/30 | Epoch Duration: 12.015 sec | Val Loss: 0.32093 | Val Acc: 0.904 |
Epoch: 19/30 | Epoch Duration: 12.026 sec | Val Loss: 0.31584 | Val Acc: 0.904 |
Epoch: 20/30 | Epoch Duration: 12.420 sec | Val Loss: 0.31596 | Val Acc: 0.905 |
Epoch: 21/30 | Epoch Duration: 12.367 sec | Val Loss: 0.32160 | Val Acc: 0.900 |
Epoch: 22/30 | Epoch Duration: 12.472 sec | Val Loss: 0.31340 | Val Acc: 0.904 |
Epoch: 23/30 | Epoch Duration: 12.134 sec | Val Loss: 0.31088 | Val Acc: 0.903 |
Epoch: 24/30 | Epoch Duration: 12.194 sec | Val Loss: 0.31267 | Val Acc: 0.903 |
Epoch: 25/30 | Epoch Duration: 12.317 sec | Val Loss: 0.31323 | Val Acc: 0.901 |
Epoch: 26/30 | Epoch Duration: 12.017 sec | Val Loss: 0.31473 | Val Acc: 0.900 |
Epoch 00026: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 27/30 | Epoch Duration: 12.051 sec | Val Loss: 0.31021 | Val Acc: 0.905 |
Epoch: 28/30 | Epoch Duration: 12.122 sec | Val Loss: 0.30899 | Val Acc: 0.904 |
Epoch: 29/30 | Epoch Duration: 11.754 sec | Val Loss: 0.30938 | Val Acc: 0.904 |
Epoch: 30/30 | Epoch Duration: 12.284 sec | Val Loss: 0.30751 | Val Acc: 0.906 |
`}
  />
  <p>
    Out of the box we get an accuracy of over <Highlight>90%</Highlight>. Think
    about how amazing those results are. We had 37 different categories, limited
    data and limited computational resources and we have essentially trained a
    linear classifier based on the features from the ResNet model. Still we get
    an accuracy of over 90%. This is the power of transfer learning.
  </p>
  <div class="separator" />
</Container>
