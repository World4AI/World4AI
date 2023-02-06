<script>
 import Container from '$lib/Container.svelte'; 
 import PythonCode from '$lib/PythonCode.svelte';
 import Alert from '$lib/Alert.svelte';
 import mnist from './mnist.png'; 

 const code1 = `import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader`;

 const code2 = `train_dataset = MNIST(root="../datasets", train=True, download=True)
test_dataset = MNIST(root="../datasets", train=False, download=False)`;

 const code3 = String.raw`print(f'A training sample is a tuple: \n{train_dataset[0]}')
print(f'There are {len(train_dataset)} training samples.')
print(f'There are {len(test_dataset)} testing samples.')
img = np.array(train_dataset[0][0])
print(f'The shape of images is: {img.shape}')`;
 const out3 = String.raw`A training sample is a tuple: 
(<PIL.Image.Image image mode=L size=28x28 at 0x7FCB9FA9E980>, 5) There are 60000 training samples.
There are 10000 testing samples.
The shape of images is: (28, 28)
`;
 const code4 = `fig = plt.figure(figsize=(10, 10))
for i in range(6):
    fig.add_subplot(1, 6, i+1)
    img = np.array(train_dataset[i][0])
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.title(f'Class Nr. {train_dataset[i][1]}')
plt.show()`;

 const code5 = `print(f'Minimum pixel value: {img.min()}')
print(f'Maximum pixel value: {img.max()}')`;

 const code6 = `import torchvision.transforms as T`;
 const code7 = `transform = T.Compose([T.PILToTensor(), 
                       T.Lambda(lambda tensor : tensor.to(torch.float32))
])`;
 const code8 = `dataset_orig = MNIST(root="../datasets/", train=True, download=True, transform=transform)`;
 const code9 = `# calculate mean and std
# we will need this part later for normalization
# we divide by 255.0, because the images will be transformed into the 0-1 range automatically
mean = (dataset_orig.data.float() / 255.0).mean()
std = (dataset_orig.data.float() / 255.0).std()`;
 const code10 = `transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])`;
 const code11 = `dataset_normalized = MNIST(root="../datasets/", train=True, download=True, transform=transform)`;
 const code12 = `# parameters
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS=10
BATCH_SIZE=32

#number of hidden units in the first and second hidden layer
HIDDEN_SIZE_1 = 100
HIDDEN_SIZE_2 = 50
NUM_LABELS = 10
ALPHA = 0.1`;
 const code13 = `dataloader_orig = DataLoader(dataset=dataset_orig, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)`;
 const code14 = `dataloader_normalized = DataLoader(dataset=dataset_normalized, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)`;

 const code15 = `def train(dataloader, model, criterion, optimizer):
    for epoch in range(NUM_EPOCHS):
        loss_sum = 0
        batch_nums = 0
        for batch_idx, (features, labels) in enumerate(dataloader):
            # move features and labels to GPU
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            # ------ FORWARD PASS --------
            probs = model(features)

            # ------CALCULATE LOSS --------
            loss = criterion(probs, labels)

            # ------BACKPROPAGATION --------
            loss.backward()

            # ------GRADIENT DESCENT --------
            optimizer.step()

            # ------CLEAR GRADIENTS --------
            optimizer.zero_grad()

            # ------TRACK LOSS --------
            batch_nums += 1
            loss_sum += loss.detach().cpu()

        print(f'Epoch: {epoch+1} Loss: {loss_sum / batch_nums}')`;

 const code16 = `class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, HIDDEN_SIZE_1),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_2, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)`;

 const code17 = `model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)
train(dataloader_orig, model, criterion, optimizer)`;

 const code18 = `model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)
train(dataloader_normalized, model, criterion, optimizer)`;
</script>

<svelte:head>
  <title>MNIST Feature Scaling - World4AI</title>
  <meta
    name="description"
    content="The MNIST handwritten classification problem is considered to be one of the introductory problems of deep learning. Yet even for such a simple task, feature scaling is a must. Luckily PyTorch makes it easy for us to apply feature scaling to computer vision tasks."
  />
</svelte:head>

<h1>Solving MNIST</h1>
<div class="separator"></div>

<Container>
  <p>It is tradition in the deep learning community to kick off the deep learning journey, by classifying handwritten digits using the <a href="http://yann.lecun.com/exdb/mnist/" rel="noreferrer" target="_blank">MNIST</a>  dataset. </p>
  <p>Additionally to the libraries, that we have used before we will utilize the <a href="https://pytorch.org/vision/stable/index.html" target='_blank' rel='noreferrer'>torchvision</a> library. Torchvision is a part of the PyTorch stack that has a lot of useful functions for computer vision. Especially useful are the datasets, like MNIST, that we can utilize out of the box without spending time collecting data on our own.</p>

  <PythonCode code={code1} />
  <p>The torchvision <code>MNIST</code> class downloads the data and returns a <code>Dataset</code> object.</p>
  <PythonCode code={code2} />
  <p>The <code>root</code> attribute designates the folder where the data will be kept. The <code>train</code> property is a boolean value. If True the object returns the train dataset, if False the object returns the test dataset. The <code>download</code> property is a boolean, which designates whether the data should be downloaded or not. You usually need to download the data only once, after that it will be cached in your root folder.</p>
  <p>Each datapoint is a tuple, consisting a PIL image and the class label. Labels range from 0 to 9, representing the correspoinding number of a handwritten digit. Images are black and white, of size 28x28 pixels. Alltogether there are 70,000 images, 60,000 training and 10,000 testing images. While this might look like a lot, modern deep learning architectures deal with millions of images. For the purpose of designing our first useful first neural network on the other hand, MNIST is the perfect dataset.</p>
  <PythonCode code={code3} />
  <PythonCode code={out3} />
  <p>Let's display some of the images to get a feel for what we are dealing with.</p>
  <PythonCode code={code4} />
  <img alt='5 MNIST images' src={mnist} />
  <p>When we look at the minimum and maximum pixel values, we will notice that they range from 0 to 255. </p>
  <PythonCode code={code5} />
  <pre class="text-sm">
Minimum pixel value: 0
Maximum pixel value: 255
</pre>  
  <p>This is the usual range that all images have. The higher the value, the higher the intensity. For black and white images 0 represents black value, 256 represents white values and all the values inbetween are shades of grey. When we start encountering colored images, we will deal with the RGB (red green blue) format. Each of the 3 so called channels (red channel, green channel and blue channel) can have values from 0 to 255. In our case we are only dealing with a single channel, because we are dealing with black and white images. So essentially an MNIST image has the format (1, 28, 28) and the batch of MNIST images, given a batch size of 32, will have a shape of (32, 1, 28, 28). This format is often abbreviated as (B, C, H, W), which stands for batch size, channels, hight, width.</p>
  <p>When it comes to computer vision, PyTorch provides scaling capabilities out of the box in <code>torchvision.transforms</code>.</p>
  <PythonCode code={code6} />
  <p>When we create a dataset using the <code>MNIST</code> class, we can pass a <code>transform</code> argument. As the name suggests we can apply a transform to images, before using those values for training. For example if we use the <code>PILToTensor</code> transform, we transform the data from PIL format to a tensor format.  Torchvision provides a great number of transforms, see <a href='https://pytorch.org/vision/stable/transforms.html#' target='_blank' rel='noreferrer'>Torchvision Docs</a>, but sometimes you might want more control. For that purpose you can use <code>transforms.Lambda()</code>, which takes a Python lambda function, in which you can process images as you desire. Often you will need to apply more than one transform. For that you can concatenate transforms using <code>transform.Compose([transform1,transform2,...])</code>. Below we prepare two sets of transforms. One set contains feature scaling, the other does not. We will both apply to MNIST and compare the results.</p>
  <p>The first set of transforms first transforms the PIL image into a Tensor and then turns the Tensor into a float32 data format. Both steps are important, because PyTorch can only work with tensors and as we intend to use the GPU, float32 is required.</p>
  <PythonCode code={code7} />
  <p>Those transforms do not include any form of scaling, therefore we expect the training to be relatively slow.</p>
  <PythonCode code={code8} />
  <p>Below we calculate the mean and the standard deviation of the images pixel values. You will notice that there is only one mean and std and not 784 (28*28 pixels). That is because in computer vision the scaling is done per channel and not per pixel. If we were dealing with color images, we would have 3 channels and would therefore require 3 mean and std calculations.</p>
  <PythonCode code={code9} />
  <p>The second set of transforms first applies <code>transforms.ToTensor</code> which turns the PIL image into a float32 Tensor and scales the image into a 0-1 range. The <code>transforms.Normalize</code> transform conducts what we call standardization or z-score normalization. The procedure essentially subracts the mean and divides by the standard deviation. If you have a color image with 3 channels, you need to provide a tuple of mean and std values, 1 for each channel.</p>
  <PythonCode code={code10} />
  <PythonCode code={code11} />
  <PythonCode code={code12} />
  <p>Based on the datasets we create two dataloaders: <code>dataloader_orig</code> without scaling and <code>dataloader_normalized</code> with scaling.</p>
  <PythonCode code={code13} />
  <PythonCode code={code14} />
  <p>The <code>train</code> function is the same generic function that we used in the previous PyTorch tutorials. </p>
  <PythonCode code={code15} />
  <p>The <code>Model</code> class is slighly different. Our batch has the shape (32, 1, 28, 28), but fully connected neural networks need a flat tensor of shape (31, 784). We essentially need to create a large vector out of all rows of the image. The layer <code>nn.Flatten()</code> does just that. Our output layer consists of 10 neurons this time. This is due to the fact, that we have 10 labels and we need ten neurons which are used as input into the softmax activation function. We do not explicitly define the softmax layer as part of the model, because our loss function will combine the softmax with the cross-entropy loss.</p>
  <PythonCode code={code16} />
  <p>Below we train the same model with and without feature scaling and compare the results. The <code>CrossEntropyLoss</code> criterion stacks the log softmax activation function and the cross-entropy loss. This log version of the softmax activation and the combination of the activation with the loss is useful for numerical stability. Theoretically you can explicitly add the <code>nn.LogSoftmax</code> activation to your model and use the <code>nn.NLLLoss</code>, but that is not recommended.</p>
  <PythonCode code={code17} />
  <pre class="text-sm">
Epoch: 1 Loss: 0.97742760181427
Epoch: 2 Loss: 0.7255294919013977
Epoch: 3 Loss: 0.7582691311836243
Epoch: 4 Loss: 0.6830052733421326
Epoch: 5 Loss: 0.6659824252128601
Epoch: 6 Loss: 0.6156877875328064
Epoch: 7 Loss: 0.6003748178482056
Epoch: 8 Loss: 0.5670294165611267
Epoch: 9 Loss: 0.6026986837387085
Epoch: 10 Loss: 0.5925905108451843
  </pre>
  <PythonCode code={code18} />
  <pre class="text-sm">
Epoch: 1 Loss: 0.7985861897468567
Epoch: 2 Loss: 0.2571895718574524
Epoch: 3 Loss: 0.17698505520820618
Epoch: 4 Loss: 0.1328950673341751
Epoch: 5 Loss: 0.1063883826136589
Epoch: 6 Loss: 0.08727587759494781
Epoch: 7 Loss: 0.0743139460682869
Epoch: 8 Loss: 0.06442411243915558
Epoch: 9 Loss: 0.05526750162243843
Epoch: 10 Loss: 0.047709111124277115
  </pre>
  <p>The difference is huge. Without feature scaling training is slow and the loss oscilates from time to time. Training with feature scaling on the other hand decreases the loss dramatically. </p>
  <Alert type='warning'>
    Not scaling input features is one of the many pitfalls you will encounter when you will work on your own projects. If you do not observe any progress, check if you have correctly scaled your input features.
  </Alert>
  <div class="separator"></div>
</Container>
