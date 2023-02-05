<script>
  import Container from '$lib/Container.svelte';
  import Highlight from '$lib/Highlight.svelte';
  import Alert from '$lib/Alert.svelte';
  import PythonCode from '$lib/PythonCode.svelte';

  const code1 = `from torch.utils.data import Dataset, DataLoader`;
  const code2 = `class ListDataset(Dataset):
    def __init__(self, size):
        self.data = list(range(size))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]`;
  const code3 = `dataset = ListDataset(100)
print(len(dataset))
print(dataset[42])`;
  const code4 = `class ImagesDataset(Dataset):
    def __init__(self, images_list):
        # list containing information about the image
        # "[/images/image0.jpg", "/images/image1.jpg]"
        self.images_list = images_list
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        file = self.images_list[idx]
        image = open_image(file)
        return image`;
  const code5 = `dataset = ListDataset(5)`;
  const code6 = `dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)`;
  const code7 = `for batch_num, data in enumerate(dataloader):
    print(f'Batch Nr: {batch_num+1} Data: {data}')`;
  const code8 = `dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=True)`;
  const code9 = `for batch_num, data in enumerate(dataloader):
    print(f'Batch Nr: {batch_num+1} Data: {data}')`;
  const code10 = `for epoch_num in range(2):
    for batch_num, data in enumerate(dataloader):
        print(f'Epoch Nr: {epoch_num + 1} Batch Nr: {batch_num+1} Data: {data}')`;
  const code11 = `dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=4)`;
  const code12 = `import torch
import sklearn.datasets as datasets
from torch.utils.data import DataLoader, Dataset`;
 const code13 = `# parameters
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS=10
BATCH_SIZE=1024
NUM_SAMPLES=1_000_000
NUM_FEATURES=10
ALPHA = 0.1

#number of hidden units in the first and second hidden layer
HIDDEN_SIZE_1 = 1000
HIDDEN_SIZE_2 = 500`;

  const code14 = `class Data(Dataset):
    def __init__(self):
        X, y = datasets.make_classification(
            n_samples=NUM_SAMPLES, 
            n_features=NUM_FEATURES, 
            n_informative=7, 
            n_classes=2, 
        )

        self.X = torch.from_numpy(X).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]`;

  const code15 = `dataset = Data()
dataloader = DataLoader(dataset=dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)`;

  const code16 = `def train(dataloader, model, criterion, optimizer):
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
            # detach() removes a tensor from a computational graph 
            # and cpu() move the tensor from GPU to CPU 
            loss_sum += loss.detach().cpu()

        print(f'Epoch: {epoch+1} Loss: {loss_sum / batch_nums}')`;

  const code17 = `class Module:
    
    def __init__(self, in_features, out_features):
        self.W = torch.normal(mean=0, 
                              std=0.1, 
                              size=(out_features, in_features), 
                              requires_grad=True, 
                              device=DEVICE, 
                              dtype=torch.float32)
        self.b = torch.zeros(1, 
                             out_features, 
                             requires_grad=True, 
                             device=DEVICE, 
                             dtype=torch.float32)
        self.parameters = [self.W, self.b]
                
    def __call__(self, features):
        return features @ self.W.T + self.b`;

  const code18 = `def sigmoid(z):
    return 1 / (1 + torch.exp(-z))`;

  const code19 = `class Model:
    
    def __init__(self):
        self.linear_1 = Module(NUM_FEATURES, HIDDEN_SIZE_1)
        self.linear_2 = Module(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.linear_3 = Module(HIDDEN_SIZE_2, 1)
        
    def __call__(self, features):
        x = self.linear_1(features)
        x = sigmoid(x)
        x = self.linear_2(x)
        x = sigmoid(x)
        x = self.linear_3(x)
        x = sigmoid(x)
        return x
    
    def parameters(self):
        parameters = [*self.linear_1.parameters, 
                      *self.linear_2.parameters,
                       *self.linear_3.parameters]
        return parameters`;

  const code20 = `features = torch.randn(BATCH_SIZE, NUM_FEATURES).to(DEVICE)
model = Model()
output = model(features)`;

  const code21 = `class SGDOptimizer:
    
    def __init__(self, parameters, alpha):
        self.alpha = alpha
        self.parameters = parameters
    
    def step(self):
        with torch.inference_mode():
            for parameter in self.parameters:
                parameter.sub_(self.alpha * parameter.grad)
                
    def zero_grad(self):
        with torch.inference_mode():
            for parameter in self.parameters:
                parameter.grad.zero_()`;

  const code22 = `def bce_loss(outputs, labels):
    loss =  -(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs)).mean()
    return loss`;
  
  const code23 = `model = Model()
optimizer = SGDOptimizer(model.parameters(), ALPHA)
criterion = bce_loss`;

  const code24 = `train(dataloader, model, criterion, optimizer)`;
  const code25 = `import torch.nn as nn`;
  const code26 = `class Module(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.parameter.Parameter(torch.normal(mean=0, std=0.1, 
                              size=(out_features, in_features)))
        self.b = nn.parameter.Parameter(torch.zeros(1, out_features))

    def forward(self, features):
        return features @ self.W.T + self.b`;

  const code27 = `class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = Module(NUM_FEATURES, HIDDEN_SIZE_1)
        self.linear_2 = Module(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.linear_3 = Module(HIDDEN_SIZE_2, 1)
        
    def forward(self, features):
        x = self.linear_1(features)
        x = torch.sigmoid(x)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = self.linear_3(x)
        x = torch.sigmoid(x)
        return x`;

  const code28 = `model = Model().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)`;
  const code29 = `train(dataloader, model, criterion, optimizer)`;
  const code30 = `class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(NUM_FEATURES, HIDDEN_SIZE_1)
        self.linear_2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.linear_3 = nn.Linear(HIDDEN_SIZE_2, 1)
    
    def forward(self, features):
        x = self.linear_1(features)
        x = torch.sigmoid(x)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = self.linear_3(x)
        x = torch.sigmoid(x)
        return x`;
  const code31 = `model = Model().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)`;
  const code32 = `train(dataloader, model, criterion, optimizer)`;
  const code33 = `class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(NUM_FEATURES, HIDDEN_SIZE_1),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_2, 1),
            )
    
    def forward(self, features):
        return self.layers(features)`;
  const code34 = `model = Model().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)`;
  const code35 = `train(dataloader, model, criterion, optimizer)`;
</script>

<svelte:head>
  <title>PyTorch Data, Modules, Optimizers, Losses - World4AI</title>
  <meta
    name="description"
    content="PyTorch provides Datasets, DataLoaders, Modules, Optimizers and Loss functions for an efficient and scalable code structure. Try to avoid code replications, instead try to use as many built-in functioality as possible."
  />
</svelte:head>

<h1>Data, Modules, Optimizers, Losses</h1>
<div class="separator"></div>

<Container>
  <p>In the last sections we have shown a very simple implementation of a neural network using PyTorch. In reality though PyTorch provides a lot of functionalities to make neural network training much more efficient and scalable. This section is dedicated to those functionalities.</p>
  <div class="separator"></div>
 
  <h2>Data</h2>
  <p>So far we have looked at very small datasets and were not necessarily concerned with how we would manage the data, but deep learning is dependent on lots and lots of data and we need to be able to store, manage and retrieve the data. When we retrieve the data we need to make sure, that we don't go beyond the capacity of our RAM or VRAM (Video RAM). PyTorch gives us a flexible way to deal with our data pipeline the way we see fit by providing the <code>Dataset</code> and the <code>DataLoader</code> classes.</p>
  <PythonCode code={code1}></PythonCode>
  <p>The <code>Dataset</code> object is the PyTorch representation of data. When we are dealing with real world data we subclass the <code>Dataset</code> class and overwrite the <code>__getitem__</code> and the <code>__len__</code> methods. Below we create a dataset that contains a list of numbers, the size of which depends on the size parameter in the <code>__init___</code> method. The <code>__getitem__</code> method implements the logic, which determines how the individual element of our data should be returned given only the index of data.</p>
  <PythonCode code={code2}></PythonCode>
  <p>We use the <code>ListDataset</code> to create a <code>list</code> with 100 elements from 0 to 99.</p>
  <PythonCode code={code3}></PythonCode>
  <pre class='text-sm'>
 100
 42</pre>
  <p>In practice we could for example use the Dataset to load an image for the index received in the <code>__getitem__</code> method. Below is a dummy implementation of such a Dataset.</p>
  <PythonCode code={code4}></PythonCode>
  <p>During the training process we only directly interact with the <code>DataLoader</code> object and not with the <code>Dataset</code> object. The goal of the <code>DataLoader</code> is to return data in batch sized pieces. Those batches can then be used for training or testing purposes. But what exaclty is a batch? The batch size tells us what proportion of the whole dataset is going to be used to calculate the graedients, before a single gradient descent step is taken.</p>

  <p>
    The approach of using the whole dataset to calculate the gradient is called <Highlight
      >batch</Highlight
    > gradient descent. Using the whole dataset has the advantage that we get a good
    estimation for the gradients, yet in many cases batch gradient descent is not
    used in practice. We often have to deal with datasets consisting of thousands
    of features and millions of samples. It is not possible to load all that data
    on the GPU's. Even if it was possible, it would take a lot of time to calculate
    the gradients for all the samples in order to take just a single training step.
  </p>
  <p>
    In <Highlight>stochastic</Highlight> gradient descent we introduce some stochasticity by shuffling
    the dataset randomly and using one sample at a time to calculate the
    gradient and to take a gradient descent step until we have used all samples
    in the dataset. The advantage of stochastic gradient descent is that we do not have to wait for the calculation of gradients for all samples, but in the process we lose the advantages of parallelization that we get with batch gradient descent. When we calculate the gradient based on one sample the calculation is going to be off. By iterating over the whole dataset the sum of the directions is going to move the weights and biases towards the optimum. In fact this behaviour is often seen as advantageous, because theoretically the imprecise gradient could potentially push a variable from a local minimum.
  </p>

  <p>
    <Highlight>Mini-batch</Highlight> gradient descent combines the advantages of the stochastic and
    batch gradient descent. Insdead of using one sample at a time ,several samples are utilized to calculate the gradients. Similar to the learning rate, the the mini-batch is a hyperparameter and needs to be determined by the developer. Usually the size is calculated as a power of 2, for example 32, 64, 128 and so on. You just need to remember that the batch needs to fit into the memory of your graphics card. The calculation of the gradients with mini-batches can be parallelized, because we can distribute the samples on different cores of the CPU/GPU. Additionally it has the advantage that theoretically our training dataset can be as large as we want.
  </p>
  <p>The <code>DataLoader</code> takes several arguments to control the above described details. The <code>dataset</code> argument expects a <code>Dataset</code> object that implements the <code>__init__</code> and <code>__getitem__</code> interface. The <code>batch_size</code> parameter determines the size of the mini-batch. The default value is 1, which is equal to stochastic gradient descent. The <code>shuffle</code> parameter is a boolean value, that detemines if the dataset will be shuffled at the beginning of the iteration process. The default value is <code>False</code>.</p>
  <p>Let's generate a ListDataset with just 5 elements for demonstration purposes.</p>
  <PythonCode code={code5}></PythonCode>
  <p>We generate a DataLoader that shuffles the dataset object and returns 2 samples at a time.</p>
  <PythonCode code={code6}></PythonCode>
  <p>Finally we iterate through the DataLoader and receive a batch at a time. Once only one object remains, a single element is returned.</p>
  <PythonCode code={code7}></PythonCode>
  <pre class="text-sm">Batch Nr: 1 Data: tensor([4, 0])
Batch Nr: 2 Data: tensor([3, 1])
Batch Nr: 3 Data: tensor([2])</pre>
  <p>Often we want our batches to always be of equal size. If a batch is too small the calculation of the gradient might be too noisy. To avoid that we can use the <code>drop_last</code> argument. The <code>drop_last</code> parameter removes the last batch, if it is less than <code>batch_size</code>. The argument defaults to False</p>
  <PythonCode code={code8}></PythonCode>
  <p>When we do the same exercise again, we end up with fewer iterations.</p>
  <PythonCode code={code9}></PythonCode>
  <pre class="text-sm">
Batch Nr: 1 Data: tensor([0, 2])
Batch Nr: 2 Data: tensor([3, 4])
</pre>
  <p>Each sample in the dataset is typically used several times in the training process. Each iteration over the whole dataset is called an <Highlight>epoch</Highlight>.</p>
  <Alert type='info'>
    An epoch is the time period, in which all the samples in the dataset have been iterated over and used for gradient calculations. 
  </Alert>
  <p>If we want to use several epochs in a training loop, all we have to do is to include an additional outer loop.</p>
  <PythonCode code={code10}></PythonCode>
<pre class="text-sm">
Epoch Nr: 1 Batch Nr: 1 Data: tensor([2, 1])
Epoch Nr: 1 Batch Nr: 2 Data: tensor([4, 0])
Epoch Nr: 2 Batch Nr: 1 Data: tensor([2, 4])
Epoch Nr: 2 Batch Nr: 2 Data: tensor([3, 1])
</pre>
 <p>
 Oftentiems it is useful to get the next batch of data using a separate process, while we are still in the process of calculating the gradients. The <code>num_workers</code> parameter determines the number of workers, that get the data in parallel. The default is 0, which means that only the main process is used.</p>
  <PythonCode code={code11}></PythonCode>
  <p>We won't notice the speed difference using such a simple example, but the speedup with large datasets might be noticable.</p>
  <p>There are more parameters, that the <code>DataLoader</code> class provides. We are not going to cover those just yet, because for the most part the usual parameters are sufficient. We will cover the special cases when the need arises. If you are faced with a problem that requires more control, you can look at the <a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"target='_blank' rel="noreferrer">PyTorch documentation</a>.</p>
  <div class="separator" />

  <h2>Training Loop</h2>
  <p>The training loop that we implemented when we solved our circular problem works just fine, but PyTorch provides much better approaches. Once our neural network architectures get more and more complex, we will be glad that we are able to utilize a more efficient training approach.</p>
  <PythonCode code={code12}></PythonCode>
  <p>This time around we explicitly set some parameters as constants. This time around we use a much higher number of samples and neurons, to demonstrate that PyTorch is able to handle those.</p>
  <PythonCode code={code13}></PythonCode>
  <p>We create a simple classification dataset with sklearn and construct a <code>Dataset</code> object.</p>
  <PythonCode code={code14}></PythonCode>
  <PythonCode code={code15}></PythonCode>
  <p>This time around we will start by looking at the desired product, the training loop, to understand what we need in order to make our code clean, modular and scalable. Instead of calculating one layer after another we will calculate our forward pass using a single call to the <code>model</code>. The model will contain all the matrix multiplications and activation functions needed to predict the probability that the features belong to a certain class. The <code>criterion</code> is essentially a loss function, in our case it is the binary cross-entropy. The <code>optimizer</code> loops through all parameters of the model and applies gradient descent when we call <code>optimizer.step</code> and clears all the gradients when we call <code>optimizer.zero_grad()</code>.</p>
  <PythonCode code={code16}></PythonCode>
  <p>In order to make our calculations more modular, we will create a <code>Module</code> class. You can think about a module as a piece of a neural network. Usually modules are those pieces of a network, that we use over and over again. In essence you create a neural network by defining and stacking modules. As we need to apply affine transformations several times, we put the logic of a linear layer into a separate class and we call that class <code>Module</code>. This module initializes a weight matrix  and a bias vector. For easier access at a later point we create an attribute <code>parameters</code>, which is just a list holding the weights and biases. We also implement the <code>__call__</code> method, which contains the logic for the forward pass.</p>
  <PythonCode code={code17}></PythonCode>
  <p>Our model needs an activation function, so we implement a sigmoid function.</p>
  <PythonCode code={code18}></PythonCode>
  <p>The <code>Model</code> class is the abstraction of the neural network. We will need three fully connected layers, so the model initializes three linear modules. In the <code>__call__</code> method we implement forward pass of the neural network. So when we call <code>model(features)</code>, the features are processed by the neural network, until the last layer is reached. Additionally we implement the <code>parameters</code> method, which returns the full list of the parameters of the model.</p>
  <PythonCode code={code19}></PythonCode>
  <p>Below we test the forward pass with random numbers. Applying the forward pass of a predefined model should feel more intuitive than our previous implementations.</p>
  <PythonCode code={code20}></PythonCode>
  <p>The optimizer class is responsible for applying gradient descent and for clearing the gradients. Ours is a simple implementation of stochastic (or batch) gradient descent, but PyTorch has many more implementations. We will study those in future chapters. Our optimizer class needs the learning rate (alpha) and the parameters of the model. When we call <code>step()</code> we loop over all parameters and apply gradient descent and when we call <code>zero_grad()</code> we clear all the gradients. Notice that the optimizer logic works independent of the exact architecture of the model, making the code more managable.</p>
  <PythonCode code={code21}></PythonCode>
  <p>Finally we implement the loss function. Once again the calculation of the loss is independent of the model or the optimizer. When we change one of the components, we do not introduce any breaking changes. If we replace the cross-entropy by mean squared error, our training loop will still keep working.</p>
  <PythonCode code={code22}></PythonCode>
  <p>Now we have all components, that are required by our training loop.</p>
  <PythonCode code={code23}></PythonCode>
  <PythonCode code={code24}></PythonCode>
<pre class="text-sm">
Epoch: 1 Loss: 0.44153448939323425
Epoch: 2 Loss: 0.26614147424697876
Epoch: 3 Loss: 0.1991310715675354
Epoch: 4 Loss: 0.16552086174488068
Epoch: 5 Loss: 0.14674726128578186
Epoch: 6 Loss: 0.13339845836162567
Epoch: 7 Loss: 0.12402357161045074
Epoch: 8 Loss: 0.11728055775165558
Epoch: 9 Loss: 0.11224914342164993
Epoch: 10 Loss: 0.1082562804222107
</pre>
  <p>You can probaly guess, that PyTorch provides classes and functions, that we implemented above, out of the box. The PyTorch module <code>torch.nn</code> contains most of the classes and functions, that we will require. </p>
  <PythonCode code={code25}></PythonCode>
  <p>When we write custom PyTorch modules we need to subclass <code>nn.Module</code>. We need to putall trainable parameters into the <code>nn.parameter.Parameter()</code> class. This tells PyTorch to put those tensors into the parameters list (which is used by the optimizer) and the tensors are automatically tracked for gradient computation. Instad of defining <code>__call__</code> as we did before, we define the <code>forward</code> method. PyTorch calls <code>forward</code>  automatically, when we call the module object. You must never call this method directly, as PyTorch does additional calculations during the forward pass, so instead of using <code>module.forward(features)</code> use <code>module(features)</code>.</p>
  <PythonCode code={code26}></PythonCode>
  <p>The great thing about PyTorch modules is their composability. Earlier created modules can be used in subsequent modules. Below for example we use the above defined <code>Module</code> class in the <code>Model</code> module. In later chapter we will see how we can create blocks of arbitrary complexity using this simple approach.</p>
  <PythonCode code={code27}></PythonCode>
  <p>PyTorch obviously provides loss functions and optimizers. We will use <code>BCELoss</code>, which calculates the binary cross-entropy loss. Optimizers are located in <code>torch.optim</code>. For now we will use stochastic gradient descent, but there are many more optimizers that we will encounter soon. </p>
  <PythonCode code={code28}></PythonCode>
  <PythonCode code={code29}></PythonCode>
  <pre class="text-sm">
Epoch: 1 Loss: 0.4358866512775421
Epoch: 2 Loss: 0.26300883293151855
Epoch: 3 Loss: 0.1951223760843277
Epoch: 4 Loss: 0.16517716646194458
Epoch: 5 Loss: 0.14785249531269073
Epoch: 6 Loss: 0.1351807564496994
Epoch: 7 Loss: 0.12569186091423035
Epoch: 8 Loss: 0.11819736659526825
Epoch: 9 Loss: 0.11242685467004776
Epoch: 10 Loss: 0.10799615830183029
  </pre>
  <p>PyTorch provides a lot of modules out of the box. An affine/linear transformation layer is a common procedure, therefore you should use <code>nn.Linear</code> instead of implementing your solutions from scratch.</p>
  <PythonCode code={code30}></PythonCode>
  <PythonCode code={code31}></PythonCode>
  <PythonCode code={code32}></PythonCode>
<pre class="text-sm">
Epoch: 1 Loss: 0.46121323108673096
Epoch: 2 Loss: 0.345653235912323
Epoch: 3 Loss: 0.26799750328063965
Epoch: 4 Loss: 0.20885568857192993
Epoch: 5 Loss: 0.16782595217227936
Epoch: 6 Loss: 0.14582592248916626
Epoch: 7 Loss: 0.1313050240278244
Epoch: 8 Loss: 0.12312141805887222
Epoch: 9 Loss: 0.11707331985235214
Epoch: 10 Loss: 0.11287659406661987
</pre>
  <p>To finish this chapter let us discuss an additional PyTorch convenience.You might have noticed, that all modules and activation functions are called one after another, where the output of one module (or activation) is used as the input into the next. In that case we can pack all modules and activations into a <code>nn.Sequential</code> object. When we call that object, the components will be executed in a sequential order.</p>
  <PythonCode code={code33}></PythonCode>
  <PythonCode code={code34}></PythonCode>
  <PythonCode code={code35}></PythonCode>
 <pre class="text-sm">
Epoch: 1 Loss: 0.4605180025100708
Epoch: 2 Loss: 0.3372548818588257
Epoch: 3 Loss: 0.27341559529304504
Epoch: 4 Loss: 0.22028055787086487
Epoch: 5 Loss: 0.17632894217967987
Epoch: 6 Loss: 0.15047569572925568
Epoch: 7 Loss: 0.1337045431137085
Epoch: 8 Loss: 0.12339214235544205
Epoch: 9 Loss: 0.11565018445253372
Epoch: 10 Loss: 0.11087213456630707
 </pre>
  <div class="separator" />
</Container>

