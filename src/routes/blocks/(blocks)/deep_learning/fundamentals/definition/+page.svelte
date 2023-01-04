<script>
  import Highlight from "$lib/Highlight.svelte";
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";

  import Table from "$lib/base/table/Table.svelte";
  import TableHead from "$lib/base/table/TableHead.svelte";
  import TableBody from "$lib/base/table/TableBody.svelte";
  import Row from "$lib/base/table/Row.svelte";
  import DataEntry from "$lib/base/table/DataEntry.svelte";
  import HeaderEntry from "$lib/base/table/HeaderEntry.svelte";

  import Neuron from "../_dl_definition/Neuron.svelte";
  import NeuralNetwork from "../_dl_definition/NeuralNetwork.svelte";
  import Layers from "../_dl_definition/Layers.svelte";

  const header = ["Location", "Size", "Price"];
  const data = [
    ["100, 100", "100", 1000000],
    ["120, 100", "120", 1200000],
    ["10, 30", "90", 200000],
    ["20, 25", "45", 110000],
  ];

  const header2 = ["Location Constructed", "Size", "Price"];
  const data2 = [
    ["City Center", "100", 1000000],
    ["City Center", "120", 1200000],
    ["City Outskirts", "90", 200000],
    ["City Outskirts", "45", 110000],
  ];
</script>

<svelte:head>
  <title>Deep Learning Definition - World4AI</title>
  <meta
    name="description"
    content="Deep learning is a branch of machine learning that utilizes deep neural nets which are able to automatically learn useful representations."
  />
</svelte:head>

<Container>
  <h1>Deep Learning</h1>
  <div class="separator" />

  <p>
    If deep learning is a subset of machine learning, we need to ask ourselves
    the following question. What makes a machine learning algorithm a deep
    learning algorithm?
  </p>
  <Alert type="info">
    <Highlight>Neural networks</Highlight>, a <Highlight
      >"deep" architecture</Highlight
    > and <Highlight>representation learning</Highlight> are all traits of deep learning.
  </Alert>
  <div class="separator" />

  <h2>Neural Networks</h2>
  <p>
    Deep learning is exclusively based on artificial neural networks. Machine
    learning algorithms that do not utilize neural networks can therefore not
    qualify as deep learning.
  </p>
  <p>
    In machine learning an artificial neuron is just a computational unit. This
    unit receives some inputs (e.g features of a house) and predicts an output
    (e.g. the price of the house) using its' model.
  </p>
  <Neuron />
  <p>
    The model that transforms features into label predictions is learned from
    data. In that sense a neuron is not different from any other machine
    learning algorithm. In fact the model of the neuron is extremely simple. For
    the most part it involves addition and multiplication. Why then would we use
    artificial neurons to build systems that should be capable of image
    recognition, text generation and other fairly complex tasks? We have the
    ability to stack neurons and thus creating a network of artificial neurons.
  </p>
  <Alert type="info">
    An artificial neural network is a set of interconnected neurons, where the
    output of a neuron is used as the input of the next neuron.
  </Alert>
  <NeuralNetwork />
  <p>
    We must not forget that each neuron has its own small model under the hood.
    This allows each of the neurons to learn a solution to a different
    subproblem. This approach is called <Highlight>divide and conquer</Highlight
    >. The whole task is divided into solvable small chunks and the solution to
    those chunks constitutes the solution to the larger task. The beauty of
    neural networks lies in their ability to use "divide and conquer"
    automatically, without being explicitly told what the subproblems are.
  </p>
  <p>
    Traditional neural networks are structured in a layered architecture. Each
    neuron takes the outputs from all neurons in the previous layer as its'
    inputs. Similarly the single output of each neuron is used as an input for
    each neuron of the next layer. Yet even though the input for each of the
    neurons in the same layer are the same, the outputs are different, because
    each neuron uses a different model internally. This type of a network is
    called a <Highlight>fully connected</Highlight> neural network.
  </p>
  <Layers />
  <p>
    The very first layer in a neural network is called <Highlight
      >input layer</Highlight
    >. The input layer does not involve any calculations. It holds the features
    of the dataset and is exclusively used as the input to the neurons in the
    next layer. The last layer is called <Highlight>output layer</Highlight>.
    The intermediary layers are called <Highlight>hidden layers</Highlight>.
  </p>
  <div class="separator" />

  <h2>Deep Architecture</h2>
  <p>
    The term deep learning implies a deep architecture, which means that we
    expect a neural network to consist of at least two hidden layers. Most
    modern neural networks have vastly more layers, but historically it was
    extremely hard to train deep neural networks. More hidden layers did not
    improve the performance of a neural network automatically. On the contrary,
    more layers usually decreased the performance, because the learning
    algorithm broke down once the distance that information needs to travel
    between neurons increased beyond a certain threshhold. Luckily researchers
    found ways to deal with huge neural networks consisting of several 100 or
    even 1000 hidden layers, but you should not forget, that the success of deep
    neural networks is a relatively new phenomenon.
  </p>
  <div class="separator" />

  <h2>Representation Learning</h2>
  <p>
    Traditional machine learning relies heavily on <Highlight
      >feature engineering</Highlight
    >.
  </p>
  <Alert type="info">
    Feature engineering is the process of generating new features from less
    relevant features using human domain knowledge. This process tends to
    improve the performance of a traditional machine learning algorithms.
  </Alert>
  <p>
    Let us consider a regression task, where we try to predict the price of a
    house based on the location and the size of the house. While the location
    seems to have an impact on the price of a house, the representation of the
    location is relatively cryptic.
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
    A human expert might know that these coordinates are useful to decide how
    far from the city center the house lies. Larger distance implies a lower
    price. From those considerations the expert might decide to classify the
    coordinates into several categories that are useful for a machine learning
    algorithm, which should lead to a better prediction quality.
  </p>
  <Table>
    <TableHead>
      <Row>
        {#each header2 as colName}
          <HeaderEntry>{colName}</HeaderEntry>
        {/each}
      </Row>
    </TableHead>
    <TableBody>
      {#each data2 as row}
        <Row>
          {#each row as cell}
            <DataEntry>{cell}</DataEntry>
          {/each}
        </Row>
      {/each}
    </TableBody>
  </Table>
  <p>
    Deep learning on the other hand must not involve any active feature
    engineering. Due to its hierarchical (layered) nature, deep neural networks
    are able to learn useful representations (hidden features) of input
    variables on their own, provided we have a large enough dataset. We can for
    example imagine that the first layers are responsible for learning those
    representations (e.g. city center), while the latter layers are responsible
    for the calculation of targets (e.g. price).
  </p>
  <div class="separator" />

  <h2>Usecase for Deep Learning</h2>
  <p>
    Deep learning has overtaken other machine learning methods in almost all
    domains. Computer vision, speach recognition and many more tasks require
    deep learning, but there are some prerequisites that need to be met if we
    want to apply deep learning.
  </p>
  <p>
    For once deep learning needs massive amounts of data if we want to achieve
    decent results. Modern image recognition systems are trained on millions of
    images and text translation systems are usually trained on all of Wikipedia.
  </p>
  <p>
    This amount of data needs to be incorporated into the training process,
    which in turn requires the use of modern graphics cards, which can be
    extremely costly. Moreover the electricity bill will cost you a small (or a
    big) fortune, because these algorithms are usually trained for several days
    at a time.
  </p>
  <p>
    The good news is, that we can utilize smaller datasets to learn how deep
    learning works. We might not produce state of the art results, but the
    knowledge should be transferable. Additionally we will discuss free/cheap
    computational resources in a different chapter, which will allows us to
    train decent models even without having access to a local graphics card.
  </p>
  <div class="separator" />
</Container>
