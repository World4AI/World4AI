<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
  import Alert from "$lib/Alert.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  const sentence1 = "Charles III is the king of the United Kingdom.";
  const sentence2 = "Queen Elizabeth II ruled for 70 years.";

  let newSentence1 = sentence1
    .toLowerCase()
    .replace(/[.]/g, "")
    .replace(/iii/g, "<unk>");
  let newSentence2 = sentence2
    .toLowerCase()
    .replace(/[.]/g, "")
    .replace(/ii/g, "<unk>");

  let vocabulary = [].concat(newSentence1.split(" "), newSentence2.split(" "));
  vocabulary = new Set(vocabulary);
  vocabulary = ["<pad>", ...vocabulary];
  vocabulary.sort((a, b) => {
    if (a === "<unk>") {
      return -1;
    }
  });

  let dictionary = {};
  vocabulary.forEach((word, idx) => {
    dictionary[word] = idx;
  });
</script>

<svelte:head>
  <title>Word Embeddings - World4AI</title>
  <meta
    name="description"
    content="It is not possible to train a neural network directly on text, as a neural network requires numerical input. Before we can train language models, we have to turn text into word embeddings, dense vectors, that represent a word in vector space."
  />
</svelte:head>

<h1>Word Embeddings</h1>
<div class="separator" />

<Container>
  <p>
    In the previous sections we have learned how we can use recurrent neural
    networks to deal with sequences. Yet we still face a problem that we need to
    solve, before we can train those models on textual data.
  </p>
  <Alert type="info">
    A neural network takes only numerical values as input, while text is
    represented as a sequence of characters or words. In order to make text
    compatible with nearal networks, it needs to be transformed into a numerical
    representation. In other words: <Highlight
      >text needs to be vectorized</Highlight
    >.
  </Alert>
  <p>
    In order to get a good intuition for the vectorization process let's work
    through a dummy example. The whole time we will assume that our dataset
    consists of these two sentences.
  </p>
  <!-- original sentences -->
  <p class="bg-violet-100 text-center p-1 rounded-xl">{sentence1}</p>
  <p class="bg-red-100 text-center p-1 rounded-xl">{sentence2}</p>
  <p>
    In the first step of the transformation process we need to tokenize our
    sentences. During the tokenization process we divide the sentence into its
    atomic parts, so called tokens. While theoretically we can divide a sentence
    into a set of characters (like letters), usually a sentence is divided into
    individual words (or subwords). Tokenization can be a daunting task, so we
    will stick to the basics here.
  </p>
  <!-- split sentences -->
  <div class="flex justify-center items-start mb-2">
    <div class="flex justify-center items-center flex-col ">
      {#each sentence1.split(" ") as word}
        <span
          class="bg-violet-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{word}</span
        >
      {/each}
    </div>
    <div class="flex justify-center items-center flex-col">
      {#each sentence2.split(" ") as word}
        <span
          class="bg-red-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{word}</span
        >
      {/each}
    </div>
  </div>
  <p>
    During tokenization, the words are often also standardized, by stripping
    punctuation and turning letters into their lower case counterparts.
  </p>
  <!-- standardized words -->
  <div class="flex justify-center items-start mb-2">
    <div class="flex justify-center items-center flex-col mb-2">
      {#each sentence1.split(" ") as word}
        <span
          class="bg-violet-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{word.toLowerCase().replace(/[.]/g, "")}</span
        >
      {/each}
    </div>
    <div class="flex justify-center items-center flex-col">
      {#each sentence2.split(" ") as word}
        <span
          class="bg-red-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{word.toLowerCase().replace(/[.]/g, "")}</span
        >
      {/each}
    </div>
  </div>

  <p>
    Once we have tokenized all words, we can create a vocabulary. A vocabulary
    is the set of all available tokens. For the sentences above we will end up
    with the following vocabulary.
  </p>

  <!-- vocabulary -->
  <div class="flex justify-center items-center flex-col">
    {#each vocabulary as word}
      <div class="bg-slate-300 w-32 text-center border border-gray-400 mb-1">
        {word}
      </div>
    {/each}
  </div>
  <p>
    You have probably noticed that additionally to the tokens we have derived
    from our dataset we have also introduced {"<pad>"} and {"<unk>"}. For the
    most part the size of the sentences is going to be of different size, but if
    we want to use batches of samples, we need to standardize the length of the
    sequence. For that purpose we use padding, which means that we fill the
    shorter sentences with {"<pad>"} tokens. The token for unknown words {"<unk>"}
    is used for words that are outside of the vocabulary. This happens for example
    if the vocabulary that is built using the training dataset does not contain some
    words from the testing dataset. Additionally we often limit the size of the vocabulary
    in order to save computational power. In our example we assume that roman numerals
    are extremely rare and replace them by the special tokens.
  </p>

  <div class="flex justify-center items-start mb-2">
    <!-- standardized words with special tokens -->
    <div class="flex justify-center items-center flex-col mb-2">
      {#each sentence1.split(" ") as word}
        <span
          class="bg-violet-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{word
            .toLowerCase()
            .replace(/[.]/g, "")
            .replace(/^iii$/g, "<unk>")}</span
        >
      {/each}
    </div>
    <div class="flex justify-center items-center flex-col mb-2">
      {#each sentence2.split(" ") as word}
        <span
          class="bg-red-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{word
            .toLowerCase()
            .replace(/[.]/g, "")
            .replace(/^ii$/g, "<unk>")}</span
        >
      {/each}
      {#each Array(2) as _, idx}
        <span
          class="bg-red-100 text-center p-1 m-1 w-32 flex justify-center border border-gray-300"
          >{"<pad>"}</span
        >
      {/each}
    </div>
  </div>

  <p>In the next step each token in the vocabulary gets assigned an index.</p>
  <!-- dictionary -->
  <div class="flex justify-center items-center flex-col">
    {#each vocabulary as word, idx}
      <div class="flex text-center mb-1 border border-gray-300">
        <div class="bg-slate-300 w-28 py-1 ">
          {word}
        </div>
        <div class="bg-yellow-200 w-10 flex justify-center items-center">
          {idx}
        </div>
      </div>
    {/each}
  </div>

  <p>Next we replace all tokens in the sentence by the corresponding index.</p>
  <!-- indexed words -->
  <div class="flex justify-center items-start mb-2">
    <div class="flex justify-center items-center flex-col mb-2">
      {#each newSentence1.split(" ") as word}
        <span
          class="bg-violet-100 text-center p-1 m-1 w-14 flex justify-center border border-gray-300"
          >{dictionary[word]}</span
        >
      {/each}
    </div>
    <div class="flex justify-center items-center flex-col mb-2">
      {#each newSentence2.split(" ") as word}
        <span
          class="bg-red-100 text-center p-1 m-1 w-14 flex justify-center border border-gray-300"
          >{dictionary[word]}</span
        >
      {/each}
      {#each Array(2) as _, idx}
        <span
          class="bg-red-100 text-center p-1 m-1 w-14 flex justify-center border border-gray-300"
          >{dictionary["<pad>"]}</span
        >
      {/each}
    </div>
  </div>
  <p>
    Theoretically we have already accomplished the task of turning words into
    numerical values, but using indices as input inot the neural netowrk is
    problematic, because those indices imply that there is a ranking in the
    words. So the word with the index 2 is somehow higher than the word with the
    index 1. Instead we create so called <Highlight>one-hot vectors</Highlight>.
    These vectors have as many dimensions, as there are tokens in the
    vocabulary. For the most part the vector consists of zeros, but at the index
    that corresponds to the word in the vocabulary the value is 1. For our
    vocabulary of size 15, so we have access to 15 one-hot vectors.
  </p>
  <div class="flex justify-center items-center">
    <div>
      {#each vocabulary as word, idx}
        <div class="flex justify-center items-center">
          <span class="bg-slate-100 w-6 flex justify-center items-center"
            >{dictionary[word]}</span
          >
          {#each Array(vocabulary.length) as _, colIdx}
            {#if colIdx === dictionary[word]}
              <span class="bg-red-100 w-4 flex justify-center items-center"
                >1</span
              >
            {:else}
              <span class="bg-blue-100 w-4 flex justify-center items-center"
                >0</span
              >
            {/if}
          {/each}
        </div>
      {/each}
    </div>
  </div>

  <p>
    Our first sentence for example would correspond to a sequence of the
    following one-hot vectors.
  </p>
  <div class="flex justify-center items-center">
    <div>
      {#each newSentence1.split(" ") as word, idx}
        <div class="flex justify-center items-center">
          <span class="bg-slate-100 w-6 flex justify-center items-center"
            >{dictionary[word]}</span
          >
          {#each Array(vocabulary.length) as _, colIdx}
            {#if colIdx === dictionary[word]}
              <span class="bg-red-100 w-4 flex justify-center items-center"
                >1</span
              >
            {:else}
              <span class="bg-blue-100 w-4 flex justify-center items-center"
                >0</span
              >
            {/if}
          {/each}
        </div>
      {/each}
    </div>
  </div>

  <p>
    While we have managed to turn our sentences into vectors, this is not the
    final step. One-hot vectors are problematic, because the dimensionaly of
    vectors growth with the size of the vocabulary. We might deal with a
    vocabulary of 30,000 words, which will produce vectors of size 30,000. If we
    input those vectors directly into a recurrent neural network, the
    computation will become intractable.
  </p>
  <p>
    Instead we first turn the one-hot representation into a dense representation
    of lower dimensionality, those vectors are called <Highlight
      >embeddings</Highlight
    >.
  </p>
  <p>
    For example those embeddings might look like the following vecors. We turn
    15-dimensional sparse vectors into 4-dimensional dense vectors.
  </p>
  <div class="flex justify-center items-center">
    <div>
      {#each vocabulary as word, idx}
        <div class="flex justify-center items-center mb-1">
          <span class="bg-slate-100 w-6 flex justify-center items-center"
            >{dictionary[word]}</span
          >
          {#each Array(4) as _, colIdx}
            <span
              class="bg-red-100 w-10 flex justify-center items-center border border-b-gray-300"
              >{Math.random().toFixed(2)}</span
            >
          {/each}
        </div>
      {/each}
    </div>
  </div>
  <p>
    Theoretically such a word embedding matrix can be trained using a fully
    connected layer. Assuming that we have a 5-dimensional one-hot vector and
    that we want to turn it into a 2-dimensional word embedding, we define the
    word embedding matrix as trainable weights of the corresponding size.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \begin{bmatrix}
      w_1 & w_2 \\ 
      w_3 & w_4 \\ 
      w_5 & w_6 \\ 
      w_7 & w_8 \\ 
      w_9 & w_{10} \\ 
    \end{bmatrix}
  `}</Latex
    >
  </div>
  <p>
    When we want to obtain the embedding for a corresponding one-hot vector, we
    multiply the two.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
    \begin{bmatrix}
      1 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \begin{bmatrix}
      w_1 & w_2 \\ 
      w_3 & w_4 \\ 
      w_5 & w_6 \\ 
      w_7 & w_8 \\ 
      w_9 & w_{10} \\ 
    \end{bmatrix}
  `}</Latex
    >
  </div>
  <p>
    The multiplication will select the correct row and result in a 2-dimensional
    vector, that can be used as an input into our sequence-language model. This
    operation will be tracked by the autograd package and those weights will
    update over the time, optimizing the embedding representation.
  </p>
  <p>
    In practice all major frameworks have a dedicated embedding layer, that does
    this operation via a lookup. Instead of actually using matrix
    multiplication, this layer takes the value from the embedding, that
    corresponds to the index of the word. This is just a more efficient
    approach, but the results of the computation should be the same.
  </p>
  <p>
    The <a
      href="https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html"
      target="_blank"
      rel="noreferrer"><code>nn.Embedding</code></a
    > layer from PyTorch has two positional arguments: the first corresponds to the
    size of the vocabulary and the second corresponds to the dimension of the embedding
    vector.
  </p>
  <PythonCode
    code={`vocabulary_size = 10
embedding_dim = 4
batch_size=5
seq_len=3`}
  />
  <p>We assume that we have 5 sentences, each consisting of 3 words.</p>
  <PythonCode
    code={`sequence = torch.randint(low=0, 
                         high=vocabulary_size, 
                         size=(batch_size, seq_len))
print(sequence.shape)`}
  />
  <PythonCode code={`torch.Size([5, 3])`} isOutput={true} />
  <p>
    The embedding maps directly from one of ten indices to the 4 dimensional
    embedding and there is no need to create one-hot encodings in PyTorch.
  </p>
  <PythonCode
    code={`embedding = nn.Embedding(num_embeddings=vocabulary_size, 
                         embedding_dim=embedding_dim)
print(embedding.weight.shape)`}
  />
  <PythonCode code={`torch.Size([10, 4])`} isOutput={true} />
  <PythonCode
    code={`embeddings = embedding(sequence)
print(embeddings.shape)`}
  />
  <PythonCode code={`torch.Size([5, 3, 4])`} isOutput={true} />
  <div class="separator" />
</Container>
