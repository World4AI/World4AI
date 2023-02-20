<script>
	import Container from "$lib/Container.svelte";
	import PythonCode from "$lib/PythonCode.svelte";
	import Highlight from "$lib/Highlight.svelte";
	import Alert from "$lib/Alert.svelte";
</script>

<svelte:head>
	<title>Mixed Precision Training - World4AI</title>
	<meta
		name="description"
		content="Mixed precision training is a technique that allows deep learning researchers to train neural networks using either 32 or 16 bit precision. This allows some layers to train faster and to reduce the memory footprint of large neural networks. PyTorch allows us to use so called automatic mixed precision, which reduces the code overhead significantly."
	/>
</svelte:head>

<h1>Mixed Precision Training</h1>
<div class="separator" />

<Container>
	<p>
		In the next section we will begin looking at different CNN architectures.
		While the older architectures are relatively easy to train, more modern
		architectures require a lot of computational power. There are different ways
		to deal with those requirements, but in this section we will specifically
		focus on <Highlight>mixed precision traing</Highlight>.
	</p>
	<p>
		So far when we trained neural networks, we utilized the <code
			>torch.float32</code
		>
		datatype. But there are layers, like linear layers and convolutions, that can
		be executed much faster using the lower
		<code>torch.float16</code> precision.
	</p>
	<Alert type="info">
		Mixed precision training allows us to train a neural network utilizing
		different levels of precision for different layers.
	</Alert>
	<p>Mixed precision training has at least two advantages.</p>
	<ol class="list-decimal list-inside">
		<li class="mb-2">
			Some layers are faster with <code>torch.float16</code> precision, therefore
			the whole training process will be significantly faster
		</li>
		<li>
			Operations using <code>torch.float16</code> require less memory than `torch.float32`
			operations. That will reduce the necessary vram requirements and will allow
			us to use a larger batch size.
		</li>
	</ol>
	<p>
		PyTorch provides a so called <Highlight>automatic mixed precision</Highlight
		> functionality, that automatically decides which of the operations will run
		with which precision. We do not have to make any of those decisions manually.
		The official PyTorch
		<a
			href="https://pytorch.org/docs/stable/amp.html"
			target="_blank"
			rel="noreferrer">documentation</a
		> provides more info on the topic.
	</p>
	<p>
		We will demonstrate the performance boost from mixed precision training with
		the help of the MNIST dataset.
	</p>
	<PythonCode
		code={`import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as T

import time`}
	/>
	<PythonCode code={`assert torch.cuda.is_available()`} />
	<PythonCode
		code={`train_dataset = MNIST(root="../datasets", train=True, download=True, transform=T.ToTensor())`}
	/>
	<PythonCode
		code={`train_dataloader=DataLoader(dataset=train_dataset, 
                            batch_size=256, 
                            shuffle=True, 
                            drop_last=True,
                            num_workers=2)`}
	/>
	<p>
		We use a much larger network, than what is required to get a good
		performance for MINST in order to demonstrate the potential of mixed
		precision training.
	</p>
	<PythonCode
		code={`cfg = [[1, 32, 3, 1, 1],
       [32, 64, 3, 1, 1],
       [64, 64, 2, 2, 0],
       [64, 128, 3, 1, 1],
       [128, 128, 3, 1, 1],
       [128, 128, 3, 1, 1],
       [128, 128, 2, 2, 0],
       [128, 256, 3, 1, 1],
       [256, 256, 2, 1, 0],
       [256, 512, 3, 1, 1],
       [512, 512, 3, 1, 1],
       [512, 512, 3, 1, 1],
       [512, 512, 2, 2, 0],
       [512, 1024, 3, 1, 1],
]

class BasicBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(num_features=kwargs['out_channels']),
            nn.ReLU()
        )
  
    def forward(self, x):
        return self.block(x)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.features = self._build_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=10),
        )
  
    def _build_layers(self, cfg):
        layers = []
        for layer in cfg:
            layers += [BasicBlock(in_channels=layer[0],
                                   out_channels=layer[1],
                                   kernel_size=layer[2],
                                   stride=layer[3],
                                   padding=layer[4])]
        return nn.Sequential(*layers)
  
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x`}
	/>
	<PythonCode
		code={`NUM_EPOCHS=10
LR=0.0001
DEVICE = torch.device('cuda')`}
	/>
	<p>
		We start by training the neural network in a familiar manner, measuring the
		time an epoch takes. We can use those values as a benchmark.
	</p>
	<PythonCode
		code={String.raw`def train(data_loader, model, optimizer, criterion):
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        losses = []
        for img, label in data_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(img)
            loss = criterion(prediction, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        end_time = time.time()
        s = f'Epoch: {epoch+1}, ' \
          f'Loss: {sum(losses)/len(losses):.4f}, ' \
          f'Elapsed Time: {end_time-start_time:.2f}sec'
        print(s)`}
	/>
	<PythonCode
		code={`model = Model(cfg)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)`}
	/>
	<p>Each epoch takes slightly over 20 seconds to complete.</p>
	<PythonCode code={`train(train_dataloader, model, optimizer, criterion)`} />
	<PythonCode
		isOutput={true}
		code={`Epoch: 1, Loss: 0.2528, Elapsed Time: 22.82sec
Epoch: 2, Loss: 0.0316, Elapsed Time: 21.99sec
Epoch: 3, Loss: 0.0201, Elapsed Time: 22.11sec
Epoch: 4, Loss: 0.0155, Elapsed Time: 22.15sec
Epoch: 5, Loss: 0.0123, Elapsed Time: 22.14sec
Epoch: 6, Loss: 0.0106, Elapsed Time: 22.18sec
Epoch: 7, Loss: 0.0112, Elapsed Time: 22.11sec
Epoch: 8, Loss: 0.0084, Elapsed Time: 22.15sec
Epoch: 9, Loss: 0.0083, Elapsed Time: 22.17sec
Epoch: 10, Loss: 0.0078, Elapsed Time: 22.14sec`}
	/>
	<p>
		We repeat the training procedure, only this time we use mixed precision
		training. For that we will utilize the <code>torch.amp</code> module. The
		<code>torch.amp.autocast</code>
		context manager runs the region below the context manager in mixed precision.
		For our purposes the forward pass and the loss are calculated using mixed precision.
		We use
		<code>torch.cuda.amp.GradScalar</code> object in order to scale the gradients
		of the loss. If the forward pass of a layer uses 16 bit precision, so will the
		backward pass. For some of the calculations the gradients will be relatively
		small and the precision of torch.float16 will not be sufficient to hold those
		small values. The values might therefore underflow. In order to remedy the problem,
		the loss is scaled and we let the scaler deal with backprop and gradient descent.
		At the end we reset the scaler object for the next batch. The three lines from
		below do exactly that.
	</p>
	<ul>
		<li><code>scaler.scale(loss).backward()</code></li>
		<li><code>scaler.step(optimizer)</code></li>
		<li><code>scaler.update()</code></li>
	</ul>
	<PythonCode
		code={String.raw`def optimized_train(data_loader, model, optimizer, criterion):
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        losses = []
        for img, label in data_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(img)
                loss = criterion(prediction, label)
            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        end_time = time.time()
        s = f'Epoch: {epoch+1}, ' \
          f'Loss: {sum(losses)/len(losses):.4f}, ' \
          f'Elapsed Time: {end_time-start_time:.2f}sec'
        print(s)`}
	/>
	<PythonCode
		code={`model = Model(cfg)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)`}
	/>
	<p>
		We improve the training speed significantly. The overhead to use automatic
		mixed precision is inconsequential when compared to the benefits.
	</p>
	<PythonCode
		code={`optimized_train(train_dataloader, model, optimizer, criterion)`}
	/>
	<PythonCode
		isOutput={true}
		code={`Epoch: 1, Loss: 0.2699, Elapsed Time: 13.00sec
Epoch: 2, Loss: 0.0319, Elapsed Time: 12.95sec
Epoch: 3, Loss: 0.0206, Elapsed Time: 12.93sec
Epoch: 4, Loss: 0.0144, Elapsed Time: 12.95sec
Epoch: 5, Loss: 0.0117, Elapsed Time: 12.95sec
Epoch: 6, Loss: 0.0104, Elapsed Time: 12.96sec
Epoch: 7, Loss: 0.0083, Elapsed Time: 12.95sec
Epoch: 8, Loss: 0.0095, Elapsed Time: 13.01sec
Epoch: 9, Loss: 0.0053, Elapsed Time: 12.97sec
Epoch: 10, Loss: 0.0091, Elapsed Time: 12.99sec`}
	/>
	<div class="separator" />
</Container>
