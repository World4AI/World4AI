<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import Latex from "$lib/Latex.svelte";
  import ButtonContainer from "$lib/button/ButtonContainer.svelte";
  import PlayButton from "$lib/button/PlayButton.svelte";
  import PythonCode from "$lib/PythonCode.svelte";

  // imports for the diagram
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Block from "$lib/diagram/Block.svelte";
  import Arrow from "$lib/diagram/Arrow.svelte";

  const references = [
    {
      author:
        "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia",
      title: "Attention is All you Need",
      journal: "Advances in Neural Information Processing Systems",
      year: "2017",
      pages: "",
      volume: "30",
      issue: "",
    },
    {
      author: "Ba, Jimmy and Kiros, Jamie and Hinton, Geoffrey",
      title: "Layer Normalization",
      year: "2016",
    },
  ];

  const sentence = ["what", "is", "your", "name"];
  const sentence2 = ["what", "is", "your", "date", "of", "birth", "?"];
  const sentence3 = ["<sos>", "what", "is", "your", "name"];
  let positionwiseFFIdxActive = 0;
  let f = () => {
    positionwiseFFIdxActive = (positionwiseFFIdxActive + 1) % 4;
  };
</script>

<svelte:head>
  <title>Transformer - World4AI</title>
  <meta
    name="description"
    content="The transformer architecture has produced state of the art results in natural language processing, computer vision and much more. Unlike the recurrent neural net, the transformer does not rely on recurrence, instead it relies on a self-attention mechanism."
  />
</svelte:head>

<Container>
  <h1>Transformer</h1>
  <div class="separator" />
  <p>
    In the year 2017 researchers from Google introduced the so called
    <Highlight>Transformer</Highlight><InternalLink type="reference" id={1} />.
    Transformers have taken the world by storm after their initial release,
    starting with NLP first and slowly but surely spilling into computer vision,
    reinforcement learning and other domains. Nowadays transformers dominate
    most deep learning research and are an integral part of most state of the
    art models.
  </p>
  <div class="separator" />

  <h2>Encoder and Decoder</h2>
  <p>
    The original paper introduced the transformer as a language translation
    tool. Similar to recurrent seq-to-seq models the transformer is structured
    as an encoder-decoder architecture. The encoder takes the original sentence,
    processes each word in a series of layers and passes the results to the
    decoder, which in turn produces a translated version of the input sentence.
  </p>
  <SvgContainer maxWidth="500px">
    <svg viewBox="0 0 400 600">
      <Block
        x={330}
        y={30}
        width={100}
        height={25}
        text={"Softmax"}
        fontSize={15}
        color="none"
      />
      <Block
        x={330}
        y={100}
        width={100}
        height={25}
        text={"Linear"}
        fontSize={15}
        color="none"
      />
      <Block
        x={330}
        y={295}
        width={100}
        height={250}
        text={"Decoder"}
        fontSize={20}
        class="fill-blue-200"
      />
      <Block
        x={70}
        y={330}
        width={100}
        height={180}
        text={"Encoder"}
        fontSize={20}
        class="fill-violet-200"
      />
      <Block
        x={330}
        y={485}
        width={100}
        height={25}
        text={"Embedding"}
        fontSize={15}
        color="none"
        class="fill-slate-300"
      />
      <Block
        x={70}
        y={485}
        width={100}
        height={25}
        text={"Embedding"}
        fontSize={15}
        color="none"
        class="fill-slate-300"
      />
      <Block
        x={330}
        y={580}
        width={100}
        height={25}
        text={"Target Text"}
        fontSize={15}
        color="none"
        class="fill-yellow-100"
      />
      <Block
        x={70}
        y={580}
        width={100}
        height={25}
        text={"Source Text"}
        fontSize={15}
        color="none"
        class="fill-yellow-100"
      />

      <Block
        x={250}
        y={295}
        width={30}
        height={30}
        text={"Nx"}
        fontSize={20}
        class="fill-blue-200"
      />
      <Block
        x={150}
        y={330}
        width={30}
        height={30}
        text={"Nx"}
        fontSize={20}
        class="fill-violet-200"
      />

      <Arrow
        data={[
          { x: 70, y: 565 },
          { x: 70, y: 505 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
      <Arrow
        data={[
          { x: 330, y: 565 },
          { x: 330, y: 505 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
      <Arrow
        data={[
          { x: 70, y: 470 },
          { x: 70, y: 430 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
      <Arrow
        data={[
          { x: 330, y: 470 },
          { x: 330, y: 430 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
      <Arrow
        data={[
          { x: 330, y: 165 },
          { x: 330, y: 120 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
      <Arrow
        data={[
          { x: 330, y: 80 },
          { x: 330, y: 50 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
      <Arrow
        data={[
          { x: 70, y: 230 },
          { x: 70, y: 200 },
          { x: 270, y: 200 },
        ]}
        strokeWidth="2"
        dashed={true}
        strokeDashArray="6 4"
        moving={true}
      />
    </svg>
  </SvgContainer>
  <p>
    The source text and the output text are embedded by their individual
    embedding layers, before they are transferred to the encoder and decoder
    respectively. We depict the encoder slightly smaller, due to a somewhat more
    complex nature of the decoder, but the components of the encoder and the
    decoder are actually almost identical. The Nx to the right of the encoder
    and to the left of the decoder indicate that both blocks are actually made
    up of several stacked layers. In the original paper 6 encoder and 6 decoder
    layers were utilized.
  </p>
  <div class="separator" />

  <h2>Embeddings</h2>
  <p>
    When we use a recurrent net, the relative position of the word in a sentence
    is implicitly conveyed to the network, because the words are processed in an
    ordered fashion. A transformer on the other hand processes all words in a
    sentence at the same time, without caring for the relative position of the
    word. Yet the order in which a word appears in a sentence does matter for
    the meaning of that sentece. We need to somehow inject addtioinal positional
    information into the embeddings.
  </p>
  <p>
    For that purpose we will use an additional embedding layer. We define an
    embedding layer which has as many embeddings, as the maximal sentence
    lengths requires. If you expect the longest sentence to consist of 100
    tokens, you will need to encode 100 values. The first token in the sentence
    will get an embedding that corresponds to index 0, the second word the
    embedding that corresponds to index 1 and so on. The output of the token
    embedding and the positional embedding is a 512 dimensional vector. We add
    both values to get our final embedding.
  </p>
  <PythonCode
    code={`class Embeddings(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(
            torch.arange(0, seq_len, device=device).view(1, seq_len)
        )
        return token_embedding + position_embedding`}
  />
  <div class="separator" />

  <h2>Attention</h2>
  <p>
    The type of attention that the transformer uses is called <Highlight
      >self-attention</Highlight
    >. Given a sequence of tokens, each token focuses on all parts of the
    sequence at the same time (including itself), but with different levels of
    attention, called attention weights.
  </p>
  <SvgContainer maxWidth={"500px"}>
    <svg viewBox="0 0 500 300">
      {#each sentence2 as word, idx}
        <Block
          x={50}
          y={25 + idx * 40}
          width={90}
          height={30}
          fontSize={20}
          text={word}
          class="fill-purple-200"
        />
        <Block
          x={450}
          y={25 + idx * 40}
          width={90}
          height={30}
          fontSize={20}
          text={word}
          class="fill-green-100"
        />
      {/each}
      {#each sentence2 as _, idx1}
        {#each sentence2 as _, idx2}
          <Arrow
            data={[
              { x: 100, y: 25 + idx1 * 40 },
              { x: 400, y: 25 + idx2 * 40 },
            ]}
            strokeWidth={2}
            moving={true}
            speed={50}
            dashed={true}
            showMarker={false}
            strokeDashArray="4, 4"
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    We miltiply attention weights with each of the token embeddings and add up
    the results, thereby creating a new embedding, that is more aware of the
    surrounding context of the word.
  </p>
  <Alert type="info"
    >The purpose of self attention is to produce context-aware embeddings.</Alert
  >
  <p>
    The easiest way to explain what that means is to look at so called homonyms.
    Words that are written the same, but have a different meaning. Let's for
    example look the meaning of the word date.
  </p>
  <p class="text-center">
    What is your <span class="bg-blue-100 inline-block p-1">date</span> of
    <span class="bg-blue-100 inline-block p-1">birthday</span>?
  </p>
  <p class="text-center">
    The <span class="bg-red-100 inline-block p-1">date</span> is my favourite
    <span class="bg-red-100 inline-block p-1">fruit</span>.
  </p>
  <p>
    In the first sentence the word date will pay attention to itself, but also
    to birthday and will incorporate the word date and the information that
    relates to time into a single vector. In the second sentence, the word date
    will pay attention to itself and the word fruit, incorporating the
    "fruitiness" aspect into the vector of the word date.
  </p>
  <p>
    Without the self-attention mechanism we would not be able to differentiate
    between the two words, because word embeddings produce the same vector for
    the same word, without incorporating the context that surrounds the word.
    But attention is obviously also useful for words other than homonyms,
    because it allows to create an embedding for each word, that is specific to
    the exact context that the word is surrounded by.
  </p>
  <p>
    In practice the self-attention mechanism in transformers is inspired by
    information retrieval systems like database queries or search engines.
    Theses systems are based on notions of a <Highlight>query</Highlight>, a <Highlight
      >key</Highlight
    >
    and a <Highlight>value</Highlight>.
  </p>
  <SvgContainer maxWidth="500px">
    <svg viewBox="0 0 500 300">
      <g transform="translate(200, 60)">
        <Block
          x={0}
          y={-40}
          width={60}
          height={30}
          text="key"
          fontSize={20}
          class="fill-indigo-200"
        />
        <Block
          x={80}
          y={-40}
          width={90}
          height={30}
          text="value"
          fontSize={20}
          class="fill-indigo-200"
        />
        {#each Array(5) as _, idx}
          <Block
            x={0}
            y={idx * 34}
            width={60}
            height={30}
            text="key {idx + 1}"
            fontSize={20}
          />
          <Block
            x={80}
            y={idx * 34}
            width={90}
            height={30}
            text="value {idx + 1}"
            fontSize={20}
          />
        {/each}
      </g>
      <Block
        x={250}
        y={270}
        width={400}
        height={40}
        text="SELECT value WHERE key='key 1'"
        fontSize={20}
        class="fill-gray-200"
      />
      <Arrow
        data={[
          { x: 50, y: 270 },
          { x: 5, y: 270 },
          { x: 5, y: 60 },
          { x: 150, y: 60 },
        ]}
        strokeWidth={2}
      />
    </svg>
  </SvgContainer>
  <p>
    In a classical database, like the one above, it is relatively clear what
    values you will get back from your query. The value is returned, if the
    query alligns with the key. If for example we use the query "SELECT value
    WHERE key='key 1'", we should get value 1 in return.
  </p>
  <p>
    When we deal with transformers we can think about a more "fuzzy" database,
    where we don't get a single value for a query, but a weighted sum of all
    values in the database. Let's for simplicity assume, that we have only two
    entries in the database with the following vector based keys.
  </p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
  k_1 = 
  \begin{bmatrix}
    1  \\
    0  \\
    1  \\
    0  \\
  \end{bmatrix},
  k_2 = 
  \begin{bmatrix}
     0 \\
     1 \\
     1 \\
     0 \\
  \end{bmatrix}
 `}</Latex
    >
  </div>
  <p>We use the following vector based query.</p>
  <div class="flex justify-center">
    <Latex
      >{String.raw`
 q = 
  \begin{bmatrix}
    1  \\
    0  \\
    1  \\
    0  \\
  \end{bmatrix}
 `}</Latex
    >
  </div>

  <p>
    We can determine the similarity between the query and each of the keys by
    calculating the dot product and we end up with the following results.
  </p>
  <div class="flex flex-col justify-center items-center">
    <Latex>{String.raw`s_1 = q \cdot k_1 = 2`}</Latex>
    <Latex>{String.raw`s_2 = q \cdot k_2 = 1`}</Latex>
  </div>
  <p>
    The similarity between the query and the first key is larger than with the
    second key, because the query and the first key are identical. The query and
    the second key are also somewhat related, because they have identical values
    in some of the vector spots.
  </p>
  <p>
    We can use these similarity scores to calculate the attention weights, by
    using them as input into the softmax function.
  </p>
  <div class="flex justify-center">
    <Latex>{String.raw`\large w_j = \dfrac{e^{s_j}}{\sum_i e^{s_i}}`}</Latex>
  </div>
  <p>
    Finally we use attention weights to calculate the weighted sum of the values
    from the database. This is the value that you retrieve from the database. A
  </p>
  <div class="flex justify-center">
    <Latex>{String.raw`a = \sum_j w_{j}v_j`}</Latex>
  </div>
  <p>
    The transformer is loosely based on this idea. In order to calculate the
    attention the transformer takes embeddings <Latex>E</Latex> as an input. These
    can be original embeddings from the embedding layer, or outputs from a previous
    encoder/decoder layer. These embeddings are used as inputs into three different
    linear layers (without any activations), producing queries
    <Latex>Q</Latex>, keys <Latex>K</Latex> and values <Latex>V</Latex> respectively.
    Those three are used to calculate the attention <Latex>A</Latex>. As the
    queries, keys and values are all based on the same inputs we are still
    dealing with self attention, but the linear layers introduce weights, that
    make the attention mechanism more powerful.
  </p>
  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 300 200">
      <Block
        x={20}
        y={100}
        width={30}
        height={30}
        text="E"
        fontSize={20}
        class="fill-slate-200"
      />
      <Block
        x={150}
        y={16}
        width={30}
        height={30}
        text="Q"
        fontSize={20}
        class="fill-red-400"
      />
      <Block
        x={150}
        y={100}
        width={30}
        height={30}
        text="K"
        fontSize={20}
        class="fill-green-400"
      />
      <Block
        x={150}
        y={184}
        width={30}
        height={30}
        text="V"
        fontSize={20}
        class="fill-blue-400"
      />
      <Block
        x={300 - 16}
        y={100}
        width={30}
        height={30}
        text="A"
        fontSize={20}
        class="fill-yellow-300"
      />
      <Arrow
        data={[
          { x: 45, y: 100 },
          { x: 125, y: 16 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 45, y: 100 },
          { x: 125, y: 100 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 45, y: 100 },
          { x: 125, y: 184 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />

      <Arrow
        data={[
          { x: 170, y: 16 },
          { x: 260, y: 90 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 170, y: 100 },
          { x: 260, y: 100 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 170, y: 184 },
          { x: 260, y: 110 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
    </svg>
  </SvgContainer>
  <p>
    The dimensions of the three matrices are identical: (batch size, sequence
    length, embedding dimension). This allows us to calculate the attention for
    all tokens and all batches in parallel.
  </p>
  <div class="flex justify-center">
    <Latex>{String.raw`A = \text{softmax}(\dfrac{QK^T}{\sqrt{d}})V`}</Latex>
  </div>
  <p>
    The only variable that is unknown to us is <Latex>{String.raw`d`}</Latex>,
    the dimension of the key. If we are dealing with a 64 dimensional vector
    embedding for example, we have to divide the similarity by the root of 64.
    According to the authors this is done, because if the similarity between two
    vectors is too strong, the softmax might get into a region with very low
    gradients. The scaling helps to alleviate that problem. The whole expression
    above is called <Highlight>scaled dot-product attention</Highlight>.
  </p>
  <PythonCode
    code={`def attention(query, key, value, mask=None):
    scores = (query @ key.transpose(1, 2)) / torch.tensor(
        embedding_dim, device=device
    ).sqrt()
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(scores, -1)
    attention = attn_weights @ value
    return attention
`}
  />
  <p>
    In the code snippet above we additionally use a so called attention mask.
    The mask is used when we want the transformer to ignore a certain part of
    the sentence. If the values of the mask amount to 0, we replace the scores
    by a value of minus infinity, which essentially amounts to attention weights
    of 0 due to the softmax.
  </p>
  <p>
    There is still one caveat we need to discuss. Instead of calculating a
    single attention <Latex>A</Latex>, we calculate a so called <Highlight
      >multihead attention</Highlight
    >. A single <Highlight>attention head</Highlight> calculates a separate <Latex
      >Q</Latex
    >, <Latex>K</Latex> and <Latex>V</Latex>, but with a reduced embedding
    dimensionality. Instead of full 512 dimensional embeddings, each head uses
    only 64 dimensional vectors. Alltogether the transformer uses 8 heads, wich
    are concatenated in the final step.
  </p>

  <SvgContainer maxWidth="400px">
    <svg viewBox="0 0 300 200">
      <Block
        x={20}
        y={100}
        width={30}
        height={30}
        text="E"
        fontSize={20}
        class="fill-slate-200"
      />
      <Block
        x={150}
        y={16}
        width={20}
        height={20}
        text="Q"
        fontSize={20}
        class="fill-red-400"
      />
      <Block
        x={160}
        y={26}
        width={20}
        height={20}
        text="Q"
        fontSize={20}
        class="fill-red-400"
      />
      <Block
        x={170}
        y={36}
        width={20}
        height={20}
        text="Q"
        fontSize={20}
        class="fill-red-400"
      />
      <Block
        x={150}
        y={100}
        width={20}
        height={20}
        text="K"
        fontSize={20}
        class="fill-green-400"
      />
      <Block
        x={160}
        y={110}
        width={20}
        height={20}
        text="K"
        fontSize={20}
        class="fill-green-400"
      />
      <Block
        x={170}
        y={120}
        width={20}
        height={20}
        text="K"
        fontSize={20}
        class="fill-green-400"
      />

      <Block
        x={150}
        y={184}
        width={20}
        height={20}
        text="V"
        fontSize={20}
        class="fill-blue-400"
      />
      <Block
        x={160}
        y={174}
        width={20}
        height={20}
        text="V"
        fontSize={20}
        class="fill-blue-400"
      />
      <Block
        x={170}
        y={164}
        width={20}
        height={20}
        text="V"
        fontSize={20}
        class="fill-blue-400"
      />
      <Block
        x={300 - 16}
        y={100}
        width={20}
        height={20}
        text="A"
        fontSize={20}
        class="fill-yellow-300"
      />
      <Block
        x={300 - 16 - 10}
        y={110}
        width={20}
        height={20}
        text="A"
        fontSize={20}
        class="fill-yellow-300"
      />
      <Block
        x={300 - 16 - 20}
        y={120}
        width={20}
        height={20}
        text="A"
        fontSize={20}
        class="fill-yellow-300"
      />

      <Arrow
        data={[
          { x: 45, y: 100 },
          { x: 125, y: 16 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 45, y: 100 },
          { x: 125, y: 100 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 45, y: 100 },
          { x: 125, y: 184 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />

      <Arrow
        data={[
          { x: 170, y: 16 },
          { x: 260, y: 90 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 170, y: 100 },
          { x: 260, y: 100 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 170, y: 184 },
          { x: 260, y: 110 },
        ]}
        strokeWidth="2"
        moving={true}
        speed={50}
        dashed={true}
        strokeDashArray="4 4"
      />
    </svg>
  </SvgContainer>
  <p>
    This procedure might be useful, because each head can learn to focus on a
    separate context, thereby improving the performance of the transformer.
  </p>
  <PythonCode
    code={`class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)

    def forward(self, query, key, value, mask=None):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        return attention(query, key, value, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(num_heads)])
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        x = [head(query, key, value, mask) for head in self.heads]
        x = torch.cat(x, dim=-1)
        x = self.dropout(self.output(x))
        return x`}
  />
  <div class="separator" />

  <h2>Position-wise Feed-Forward Networks</h2>
  <p>
    The encoder and decoder apply a so called position-wise feed-forward neural
    network. In essence that means that the same network, with the same weights
    is applied to each position of the sentence individually. Each embedded word
    in the sequence is passed though the network without interacting with any
    other word.
  </p>
  <p>
    The position-wise network is a two-layer neural network, that takes an
    embedding of size 512, increases the dimensionality to 2048 in the first
    linear layer, applies a ReLU activation function, followed again by a linear
    layer that transforms the embeddings back to lengths 512.
  </p>
  <ButtonContainer><PlayButton {f} /></ButtonContainer>
  <SvgContainer maxWidth={"600px"}>
    <svg viewBox="0 0 500 250">
      {#each sentence as word, idx}
        <!-- words -->
        <Block
          x={70 + 120 * idx}
          y={230}
          width={90}
          height={30}
          fontSize={20}
          text={word}
          class="fill-blue-100"
        />
        <!-- embeddings -->
        {#each Array(5) as _, embeddingIdx}
          {#if idx === positionwiseFFIdxActive}
            <Arrow
              data={[
                { x: 40 + 120 * idx + embeddingIdx * 15, y: 195 },
                { x: 220 + embeddingIdx * 15, y: 130 },
              ]}
              dashed={true}
              strokeDashArray="2 2"
            />
          {/if}
          <Block
            x={40 + 120 * idx + embeddingIdx * 15}
            y={200}
            width={10}
            height={10}
            fontSize={20}
            text={""}
            class="fill-blue-100"
          />
        {/each}
      {/each}
      <!-- neural network -->
      {#each Array(5) as _, neuronIdx1}
        <Block
          x={220 + neuronIdx1 * 15}
          y={125}
          width={10}
          height={10}
          fontSize={20}
          text={""}
          class="fill-red-400"
        />
        <Block
          x={220 + neuronIdx1 * 15}
          y={15}
          width={10}
          height={10}
          fontSize={20}
          text={""}
          class="fill-red-400"
        />
      {/each}
      {#each Array(8) as _, neuronIdx1}
        <Block
          x={195 + neuronIdx1 * 15}
          y={70}
          width={10}
          height={10}
          fontSize={20}
          text={""}
          class="fill-red-400"
        />
      {/each}
    </svg>
  </SvgContainer>
  <p>
    PyTorch does this procedure automatically. Each dimension of a tensor,
    except for the last one is treated similar to a batch dimension. Only the
    last dimension, the embedding dimension, is processed through the neural
    network. The batch dimensions are regarded as additional samples, which can
    be are processed simultaneouly on the GPU.
  </p>
  <PythonCode
    code={`class PWFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)`}
  />
  <div class="separator" />

  <h2>Encoder Layer</h2>
  <p>
    The encoder layer is a combination of two sublayers: a multihead attention
    and a position-wise feed-forward neural network. Both sublayers make up an
    encoder layer, that is stacked N times.
  </p>
  <SvgContainer maxWidth="500px">
    <svg viewBox="0 0 400 400">
      <Block x={200} y={200} width={200} height={350} text={""} fontSize={20} />
      <Block x={350} y={130} width={30} height={30} text={"Nx"} fontSize={20} />

      <!-- 3 arrows -->
      <Arrow
        data={[
          { x: 200, y: 400 },
          { x: 200, y: 325 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 200, y: 400 },
          { x: 200, y: 340 },
          { x: 140, y: 340 },
          { x: 140, y: 325 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 200, y: 400 },
          { x: 200, y: 340 },
          { x: 260, y: 340 },
          { x: 260, y: 325 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- first skip connection -->
      <Arrow
        data={[
          { x: 200, y: 400 },
          { x: 200, y: 360 },
          { x: 50, y: 360 },
          { x: 50, y: 250 },
          { x: 115, y: 250 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- connect attention to feed forward -->
      <Arrow
        data={[
          { x: 200, y: 300 },
          { x: 200, y: 175 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- second skip connection -->
      <Arrow
        data={[
          { x: 200, y: 200 },
          { x: 50, y: 200 },
          { x: 50, y: 100 },
          { x: 115, y: 100 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- connect fc to next layer-->
      <Arrow
        data={[
          { x: 200, y: 150 },
          { x: 200, y: 10 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- encoder components -->
      <Block
        x={200}
        y={100}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={200}
        y={150}
        width={150}
        height={30}
        text={"P.w. Feed Forward"}
        fontSize={15}
        class="fill-red-400"
      />

      <Block
        x={200}
        y={250}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={200}
        y={300}
        width={150}
        height={30}
        text={"Multihead Atention"}
        fontSize={15}
        class="fill-red-400"
      />
    </svg>
  </SvgContainer>

  <p>
    After both sublayers we use an "Add & Norm" block. The "Add" component
    indicates that we are using skip connections in order to mitigate vanishing
    gradients and stabilize training. The "Norm" part indicates, that we
    normalize the values, before we send the results to the next layer or
    sub-layer. In the original paper the authors used a so called layer
    normalization<InternalLink type="reference" id={2} />. When we use layer
    norm we do not calculate the mean and the standard deviation for the same
    features over the different batches, but over the different features within
    the same batch.
  </p>
  <p>
    Assuming we use a batch size of 5 and 10 features, the two approaches would
    differ in the following way.
  </p>
  <SvgContainer maxWidth="500px">
    <svg viewBox="0 0 270 100">
      <Block
        x={35}
        y={10}
        width={60}
        height={15}
        text="Batch Norm"
        fontSize={9}
      />
      {#each Array(5) as _, batchIdx}
        {#each Array(10) as _, featureIdx}
          <Block
            x={10 + 12 * featureIdx}
            y={40 + 12 * batchIdx}
            width={10}
            height={10}
            class={featureIdx === 0 ? "fill-red-400" : "none"}
          />
        {/each}
      {/each}

      <Block
        x={175}
        y={10}
        width={60}
        height={15}
        text="Layer Norm"
        fontSize={9}
      />
      {#each Array(5) as _, batchIdx}
        {#each Array(10) as _, featureIdx}
          <Block
            x={150 + 12 * featureIdx}
            y={40 + 12 * batchIdx}
            width={10}
            height={10}
            class={batchIdx === 0 ? "fill-red-400" : "none"}
          />
        {/each}
      {/each}
    </svg>
  </SvgContainer>
  <p>
    You will notice in practice, that many modern implementation deviate from
    the original by normalizing the values first, before they are used as inputs
    into the sublayers. This is found to work better empirically and we do the
    same in the code snippets below.
  </p>
  <PythonCode
    code={`class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadAttention()
        self.feed_forward = PWFeedForward()

    def forward(self, src, mask=None):
        normalized = self.norm1(src)
        src = src + self.self_attention(normalized, normalized, normalized, mask)
        normalized = self.norm2(src)
        src = src + self.feed_forward(normalized)
        return src


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for encoder in self.layers:
            src = encoder(src, mask)
        return src`}
  />
  <div class="separator" />

  <h2>Decoder Layer</h2>
  <p>
    The decoder layer is also stacks multihead-attention and position-wise
    feed-forward networks, but the implementation details are slightly
    different.
  </p>
  <SvgContainer maxWidth="500px">
    <svg viewBox="0 0 400 500">
      <Block x={200} y={250} width={200} height={450} text={""} fontSize={20} />
      <Block x={380} y={200} width={30} height={30} text={"Nx"} fontSize={20} />

      <!-- 3 bottom arrows -->
      <Arrow
        data={[
          { x: 200, y: 500 },
          { x: 200, y: 425 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 200, y: 500 },
          { x: 200, y: 440 },
          { x: 140, y: 440 },
          { x: 140, y: 425 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />
      <Arrow
        data={[
          { x: 200, y: 500 },
          { x: 200, y: 440 },
          { x: 260, y: 440 },
          { x: 260, y: 425 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- first skip connection -->
      <Arrow
        data={[
          { x: 200, y: 500 },
          { x: 200, y: 460 },
          { x: 50, y: 460 },
          { x: 50, y: 350 },
          { x: 115, y: 350 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- connect attention to second attention -->
      <Arrow
        data={[
          { x: 200, y: 400 },
          { x: 200, y: 300 },
          { x: 140, y: 300 },
          { x: 140, y: 275 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />
      <!-- from encoder to decoder -->
      <Arrow
        data={[
          { x: 350, y: 295 },
          { x: 200, y: 295 },
          { x: 200, y: 275 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4, 4"
      />
      <Arrow
        data={[
          { x: 350, y: 290 },
          { x: 260, y: 290 },
          { x: 260, y: 275 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4, 4"
      />

      <!-- second skip connection -->
      <Arrow
        data={[
          { x: 200, y: 320 },
          { x: 50, y: 320 },
          { x: 50, y: 200 },
          { x: 115, y: 200 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- connect attention to feed forward -->
      <Arrow
        data={[
          { x: 200, y: 250 },
          { x: 200, y: 125 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- third skip connection -->
      <Arrow
        data={[
          { x: 200, y: 170 },
          { x: 50, y: 170 },
          { x: 50, y: 50 },
          { x: 115, y: 50 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- connect fc to next layer-->
      <Arrow
        data={[
          { x: 200, y: 100 },
          { x: 200, y: 10 },
        ]}
        strokeWidth={2}
        dashed={true}
        strokeDashArray="4 4"
      />

      <!-- decoder components -->
      <Block
        x={200}
        y={50}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={200}
        y={100}
        width={150}
        height={30}
        text={"P.w. Feed Forward"}
        fontSize={15}
        class="fill-red-400"
      />

      <Block
        x={200}
        y={200}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={200}
        y={250}
        width={150}
        height={30}
        text={"Multihead Atention"}
        fontSize={15}
        class="fill-red-400"
      />

      <Block
        x={200}
        y={350}
        width={150}
        height={30}
        text={"Add & Norm"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={200}
        y={400}
        width={150}
        height={30}
        text={"Multihead Atention"}
        fontSize={15}
        class="fill-red-400"
      />
      <Block
        x={90}
        y={400}
        width={60}
        height={30}
        text={"Masked"}
        fontSize={15}
        class="fill-red-400"
      />

      <!-- encoder outputs -->
      <Block
        x={355}
        y={295}
        width={80}
        height={30}
        text={"Encoder"}
        fontSize={15}
        class="fill-slate-100"
      />
    </svg>
  </SvgContainer>
  <p>
    The embeddings from the target text are masked. This means that when we use
    multihead attention, the attention mechanism is only allowed to pay
    attention to words that were already generated. If that wouldn't be the
    case, the transformer would be allowed to cheat, by looking at the words it
    is expected to produce.
  </p>
  <SvgContainer maxWidth={"400px"}>
    <svg viewBox="0 0 200 130">
      <g transform="translate(0 -20)">
        {#each Array(4) as _, idx}
          <Block
            x={25 + idx * 50}
            y={130}
            width={30}
            height={20}
            text={sentence3[idx]}
            class="fill-lime-200"
            fontSize={9}
          />
          <Block
            x={25 + idx * 50}
            y={35}
            width={30}
            height={20}
            text={sentence3[idx + 1]}
            class="fill-yellow-200"
            fontSize={9}
          />
          {#each Array(4) as _, idx2}
            {#if idx2 >= idx}
              <Arrow
                data={[
                  { x: 25 + idx * 50, y: 120 },
                  { x: 25 + idx2 * 50 + idx * 4, y: 50 },
                ]}
                dashed={true}
                moving={true}
                speed={50}
              />
            {/if}
          {/each}
        {/each}
      </g>
    </svg>
  </SvgContainer>
  <p>
    When we are about to produce the first word, the transformer is only allowed
    to see the start of sequence token. If it is about to produce the word "is",
    it is only allowed to additionally see the word "what". The transformer can
    pay attention to the words that came before, but never future words. To
    accomplish that practically we create a mask, which contains zeros at future
    positions.
  </p>
  <p>
    You have already probably noticed, that the decoder has an additional
    attention layer. The second multi-head attention layer combines the encoder
    with the decoder. This time the queries, values and keys do not come from
    the same embeddings. The query is based on the decoder embeddings, while the
    key and the value are based on the output of the last encoder layer. This
    attention mechanism is called <Highlight>cross-attention</Highlight>.
  </p>
  <p>The rest of the implementation is similar to the encoder.</p>
  <PythonCode
    code={`class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadAttention()
        self.cross_attention = MultiHeadAttention()
        self.feed_forward = PWFeedForward()

    def forward(self, src, trg, src_mask, trg_mask):
        normalized = self.norm1(trg)
        trg = trg + self.self_attention(normalized, normalized, normalized, trg_mask)
        normalized = self.norm2(trg)
        trg = trg + self.cross_attention(trg, src, src, src_mask)
        normalized = self.norm3(trg)
        trg = trg + self.feed_forward(normalized)
        return trg


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        for decoder in self.layers:
            trg = decoder(src, trg, src_mask, trg_mask)
        return trg
`}
  />
  <div class="separator" />

  <h2>Further Sources</h2>
  <p>
    Understanding the transformer with all the details is not an easy task. It
    is unlikely that the section above is sufficient to completely cover this
    architecture. You should therefore study as many sources as possible. Up to
    this day the transformer is the most performant architecture in deep
    learning and it is essential to have a solid understanding of the basic
    principles of this architecture.
  </p>
  <p>
    You have to read the original paper by Vasvani et. al. We had to omit some
    of the implementation details, so if you want to implement the transformer
    on your own, reading this paper is a must.
  </p>
  <p>
    <a
      href="https://jalammar.github.io/illustrated-transformer/"
      target="_blank"
      rel="noreferrer">"The Illustrated Transformer"</a
    > by Jay Alamar is a great resource if you need additional intuitive illustrations
    and explanations.
  </p>
  <p>
    <a
      href="http://nlp.seas.harvard.edu/annotated-transformer/"
      target="_blank"
      rel="noreferrer">"The Annotated Transformer"</a
    > from the Harvard University is a great choice if you need an in depths PyTorch
    implementation.
  </p>
  <p>
    The book <a
      href="https://transformersbook.com/"
      target="_blank"
      rel="noreferrer"
    >
      "Natural Language Processing with Transformers"</a
    > covers theory and applications of different transformer models in a very approachable
    manner.
  </p>
</Container>

<Footer {references} />
