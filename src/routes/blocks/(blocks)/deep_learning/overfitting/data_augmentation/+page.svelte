<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  import overfitting from "./overfitting.png";
  import mnistOrig from "./mnist_orig.png";
  import mnistBlur from "./mnist_blur.png";
  import mnistFlipped from "./mnist_flipped.png";
  import mnistRotated from "./mnist_rotated.png";

  const code1 = `# function to loop over a list of images and to draw them using matplotlib
def draw_images(images, name):
    fig = plt.figure(figsize=(10, 10))
    for i, img in enumerate(images):
        fig.add_subplot(1, len(images), i+1)
        img = img.squeeze()
        plt.imshow(img, cmap="gray")
        plt.axis('off')
    plt.savefig(f'{name}.png', bbox_inches='tight')
    plt.show()`;
  const code2 = `# original images
images = [train_validation_dataset[i][0] for i in range(6)]
draw_images(images, 'minst_orig')`;
  const code3 = `# rotate
transform = T.RandomRotation(degrees=(-30, 30))
transformed_images = [transform(img) for img in images]
draw_images(transformed_images, 'mnist_rotated')`;
  const code4 = `# gaussian blur
transform = T.GaussianBlur(kernel_size=(5,5))
transformed_images = [transform(img) for img in images]
draw_images(transformed_images, 'mnist_blur')`;
  const code5 = `# flip
transform = T.RandomHorizontalFlip(p=1)
transformed_images = [transform(img) for img in images]
draw_images(transformed_images, 'mnist_flipped')`;
  const code6 = `transform = T.Compose([
    T.GaussianBlur(kernel_size=(5,5)),
    T.ToTensor(),
])`;
  const code7 = `train_validation_dataset_aug = MNIST(root="../datasets/", train=True, download=True, transform=transform)
train_dataset_aug = Subset(train_validation_dataset_aug, train_idxs)
val_dataset_aug = Subset(train_validation_dataset_aug, val_idxs)


train_dataloader_aug = DataLoader(dataset=train_dataset_aug, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)
train_dataloader_aug = DataLoader(dataset=val_dataset_aug, 
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              drop_last=False,
                              num_workers=4)`;
  const code8 = `model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=0.005)`;
  const code9 = `history = train(NUM_EPOCHS, train_dataloader_aug, train_dataloader_aug, model, criterion, optimizer)`;
  const output9 = `Epoch: 1/50|Train Loss: 0.4877 |Val Loss: 0.4859 |Train Acc: 0.8565 |Val Acc: 0.8580
Epoch: 10/50|Train Loss: 0.1616 |Val Loss: 0.1652 |Train Acc: 0.9507 |Val Acc: 0.9470
Epoch: 20/50|Train Loss: 0.1158 |Val Loss: 0.1149 |Train Acc: 0.9657 |Val Acc: 0.9633
Epoch: 30/50|Train Loss: 0.1366 |Val Loss: 0.1377 |Train Acc: 0.9578 |Val Acc: 0.9590
Epoch: 40/50|Train Loss: 0.1215 |Val Loss: 0.1187 |Train Acc: 0.9652 |Val Acc: 0.9638
Epoch: 50/50|Train Loss: 0.1265 |Val Loss: 0.1209 |Train Acc: 0.9635 |Val Acc: 0.9648
`;
  const code10 = `plot_history(history)`;
</script>

<svelte:head>
  <title>Data Augmentation - World4AI</title>
  <meta
    name="description"
    content="One of the best ways to fight overfitting is to use more data for training, we do not always posess sufficient amounts of data. Data augmentation is a simple technique to produce synthetic data that can be used to train a neural network."
  />
</svelte:head>

<h1>Data Augmentation</h1>
<div class="separator" />

<Container>
  <p>
    One of the best ways to reduce the chances of overfitting is to gather more
    data. Let's assume that we are dealing with MNIST and want to teach a neural
    net to recognize handwritten digits. If we provide the neural network with
    just ten images for training, one for each category, there is a very little
    chance, that the network will generalize and actually learn to recognize the
    digits. Instead it will memorize the specific samples. If we provide the
    network with millions of images on the other hand, the network has a smaller
    chance to memorize all those images.
  </p>
  <p>
    MNIST provides 60,000 training images and 10,000 test images. This data is
    sufficient to train a good performing neral network, because the task is
    comparatively easy. In modern day deep learning this amount of data would be
    insufficient and we would be required to collect more data. Oftentimes
    collection of additional samples is not feasable and we will resort to <Highlight
      >data augmentation</Highlight
    >.
  </p>
  <Alert type="info">
    Data augmentation is a techinque that applies transformations to the
    original dataset, thereby creating synthetic data, that can be used in
    training.
  </Alert>
  <p>
    We can for example rotate, blur or flip the images, but there are many more
    options available. You can have a look at the <a
      href="https://pytorch.org/vision/stable/transforms.html"
      rel="noreferrer"
      target="_blank">PyTorch documentation</a
    > to study the available options.
  </p>
  <p>
    It is not always the case that we would take the 60,000 MNIST training
    samples, apply let's say 140,000 transformations and end up with 200,000
    images for training. Often we apply random transformations to each batch of
    traning that we encounter. For example we could slightly rotate and blur
    each of the 32 images in our batch using some random parameters. That way
    our neural network never encounters the exact same image twice and has to
    learn to generalize. This the approach we are going to take with PyTorch.
  </p>
  <p>
    We are going to use the exact same model and training loop, that we used in
    the previous section, so let us focus on the parts that acutally change.
  </p>
  <p>We create a simple function, that saves and displays MNIST images.</p>
  <PythonCode code={code1} />
  <p>First we generate 6 non-augmented images from the training dataset.</p>
  <PythonCode code={code2} />
  <img src={mnistOrig} alt="Original MMNIST images" />
  <p>
    We can rotate the images by using <code>T.RandomRotation</code>. We use an
    angle between -30 and 30 degrees to get the following results.
  </p>
  <PythonCode code={code3} />
  <img src={mnistRotated} alt="Rotated MMNIST images" />
  <p>
    We can blur the images by using <code>T.GaussianBlur</code>.
    <PythonCode code={code4} />
    <img src={mnistBlur} alt="Blurred MMNIST images" />
  </p>
  <p>
    Or we can randomly flip the images by using <code
      >T.RandomHorizontalFlip</code
    >.
    <PythonCode code={code5} />
    <img src={mnistFlipped} alt="Flipped MMNIST images" />
  </p>
  <p>
    There are many more different augmentation transforms available, but in this
    example we will only apply one. First apply gaussian blur to the PIL image
    and then we transform the result into a PyTorch tensor.
  </p>
  <PythonCode code={code6} />
  <p>
    As we have created new transforms, we have to to create a new training
    dataset and dataloader.
  </p>
  <PythonCode code={code7} />
  <p>
    It turns out that the learning rate that we used before is too large if we
    apply augmentations, so we use a reduced learning rate.
  </p>
  <PythonCode code={code8} />
  <p>By using augmentation we reduce overfitting significantly.</p>
  <PythonCode code={code9} />
  <PythonCode code={output9} isOutput={true} />
  <p>The validation plot follows the trainig plot very closely.</p>
  <PythonCode code={code10} />
  <img src={overfitting} alt="Overfitting after augmentation" />
  <p>
    It is relatively easy to augment image data, but it is not always easy to
    augment text or time series data. To augment text data on Kaggle for
    example, in some competitions people used google translate to translate a
    sentence into a foreign language first and then translate the sentence back
    into english. The sentence changes slightly, but is similar enough to be
    used in the training process. Sometimes you might need to get creative to
    find a good data augmentation approach.
  </p>
  <p>
    Before we move on to the next section let us mention that there is a
    significantly more powerful technique to deal with limited data: <Highlight
      >transfer learning</Highlight
    >. Tranfer learning allows you to use a model, that was pretrained on
    millions of images or millions of texts, thereby allowing you to finetune
    the model to your needs. Those types of models need significantly less data
    to learn a particular task. It makes little sense to cover transfer learning
    in detail, before we have learned convolutional neural networks or
    transformers. Once we encounter those types of networks we will discuss this
    topic in more detail.
  </p>
  <div class="separator" />
</Container>
