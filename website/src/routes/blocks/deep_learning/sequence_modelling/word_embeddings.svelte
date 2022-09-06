<script>  
  import Container from "$lib/Container.svelte";
  import SvgContainer from "$lib/SvgContainer.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";

  const sentence1 = "Henry VIII was king of England.";
  const sentence2 = "Elizabeth II is queen of the United Kingdom.";

  let newSentence1 = sentence1.toLowerCase().replace(/[.]/g, "").replace(/viii/g, "<unk>")
  let newSentence2 = sentence2.toLowerCase().replace(/[.]/g, "").replace(/ii/g, "<unk>")
  
  let vocabulary = [].concat(newSentence1.split(" "), newSentence2.split(" "));
  vocabulary = new Set(vocabulary);
  vocabulary = ["<pad>", ...vocabulary];
  vocabulary.sort();

  let dictionary = {};
  vocabulary.forEach((word, idx) => {
    dictionary[word] = idx; 
  });
</script>

<h1>Word Embeddings</h1>
<div class="separator" />

<Container>
  <p>In the previous sections we have learned how we can use recurrent neural networks to learn sequence models. Yet we still face a problem that we need to solve, before we can train those models on text.</p>
  <p class="info">A neural network takes only numerical values as input, while text is represented as a sequence of characters or words. In order to make text compatible for training and inference, it needs to be transformed into a numerical representation. In other words text needs to be vectorized.</p>
  <p>In order to get a good intuition for the vectorization process let's work through a dummy example. The whole time we will assume that our dataset consists of these two sentences.</p>
  <!-- original sentences -->
  <p class="yellow text-center">{sentence1}</p>
  <p class="yellow text-center">{sentence2}</p>
  
  <p>In the first step of the transformation process we need to tokenize our sentences. During the tokenization process we divide the sentence into its atomic parts, so called tokens. While theoretically we can divide a sentence into a set of characters (like letters), usually a sentence is divided into individual words. Tokenization can be a daunting task, so we will stick to the basics here.</p>
  <!-- split sentences -->
  <div class="tokens">
    {#each sentence1.split(" ") as word}
      <span class="word">{word}</span>
    {/each}
  </div>
  <div class="tokens">
    {#each sentence2.split(" ") as word}
      <span class="word">{word}</span>
    {/each}
  </div>

  <p>During tokenization, the words are often also standardized, by stripping punctuation and turing letters into their lower case counterparts.</p>
  <!-- standardized words -->
  <div class="tokens">
    {#each sentence1.split(" ") as word}
      <span class="word">{word.toLowerCase().replace(/[.]/g, "")}</span>
    {/each}
  </div>
  <div class="tokens">
    {#each sentence2.split(" ") as word}
      <span class="word">{word.toLowerCase().replace(/[.]/g, "")}</span>
    {/each}
  </div>

  <p>Once we have tokenized all words, we can create a vocabulary. A vocabulary is the set of all available tokens. For the sentences above we will end up with the following vocabulary.</p>
  <!-- vocabulary -->
  <div class="vocabulary">
    {#each vocabulary as word}
      <div class="vocabulary-word">
        {word}
      </div>
    {/each}
  </div>
  <p>You have probably noticed that additionally to the tokens we have derived from our dataset we have also introduced {"<pad>"} and {"<unk>"}. For the most part the size of the sentences is going to be of different size, but if we want to use batches of samples, we need to standardize the length of the sequence. For that purpose we use padding ({"<pad>"}). The token for unknown words {"<unk>"} is used for words that are outside of the vocabulary. This happens for example if the vocabulary that is built using the training dataset does not contain some words from the testing dataset. Additionally we often limit the size of the vocabulary in order to save computational power. In our example we assume that roman numerals are extremely rare and replace them by the special token.</p>

  <!-- standardized words with special tokens -->
  <div class="tokens">
    {#each sentence1.split(" ") as word}
      <span class="word">{word.toLowerCase().replace(/[.]/g, "").replace(/^viii$/g, "<unk>")}</span>
    {/each}
    {#each Array(2) as _, idx}
      <span class="word">{"<pad>"}</span>
    {/each}
  </div>
  <div class="tokens">
    {#each sentence2.split(" ") as word}
      <span class="word">{word.toLowerCase().replace(/[.]/g, "").replace(/^ii$/g, "<unk>")}</span>
    {/each}
  </div>

  <p>Each token in the vocabulary gets assigned an index.</p>
  <!-- dictionary -->
  <div class="vocabulary">
    {#each vocabulary as word, idx}
      <div class="vocabulary-row">
        <div class="vocabulary-word">
          {word} 
        </div>
        <div class="vocabulary-idx">
          {idx}
        </div>
      </div>
    {/each}
  </div>

  <p>And we replace all tokens in the sentence by the corresponding index.</p>
  <!-- indexed words -->
  <div class="tokens">
    {#each newSentence1.split(" ") as word}
      <span class="index">{dictionary[word]}</span>
    {/each}
    {#each Array(2) as _, idx}
      <span class="index">{dictionary["<pad>"]}</span>
    {/each}
  </div>
  <div class="tokens">
    {#each newSentence2.split(" ") as word}
      <span class="index">{dictionary[word]}</span>
    {/each}
  </div>
  
  <p>To turn each index into a vector we use one-hot encoding. These vectors are as long as there are tokens in the vocabulary. For the most part the vector consists of zeros, but at the correct index the value is 1. For our vocabulary of size 13, we have access to 13 one-hot vectors.</p>
  <div class="tokens">
    <div>
    {#each vocabulary as word, idx}
      <div>
        <span class="index">{dictionary[word]}</span>
        {#each Array(vocabulary.length) as _, colIdx}
          {#if colIdx===dictionary[word]} 
            <span class="hot-index">1</span>
          {:else}
            <span class="cold-index">0</span>
          {/if}
        {/each}
      </div>
    {/each}
    </div>
  </div>

  <p>Our second sentence for example would correspond to a sequence of the following one-hot vectors.</p>
  <div class="tokens">
    <div>
    {#each newSentence2.split(" ") as word, idx}
      <div>
        <span class="index">{dictionary[word]}</span>
        {#each Array(vocabulary.length) as _, colIdx}
          {#if colIdx===dictionary[word]} 
            <span class="hot-index">1</span>
          {:else}
            <span class="cold-index">0</span>
          {/if}
        {/each}
      </div>
    {/each}
    </div>
  </div>

  <p>While we have managed to turn our sentences into vectors, this is not the final step. One-hot vectors are problematic, because the dimensionaly of vectors growth with the size of the vocabulary. We might deal with a vocabulary of 30,000 words, which will produce vectors of size 30,000. If we input those vectors directly into a recurrent neural network, the computation will become intractable.</p>
  <p>Instead we first turn the one-hot representation into a dense representation of lower dimensionality, those vectors are called <Highlight>word embeddings</Highlight>.</p>
  <p>Below we exemplify how those embeddings might look like. We turn 13-dimensional sparse vectors into 4-dimensional dense vectors.</p>
  <div class="tokens">
    <div>
    {#each vocabulary as word, idx}
      <div>
        <span class="index">{dictionary[word]}</span>
        {#each Array(4) as _, colIdx}
          <span class="embedding-entry">{Math.random().toFixed(2)}</span>
        {/each}
      </div>
    {/each}
    </div>
  </div>
  <p>Theoretically such a word embedding matrix can be trained using a fully connected layer. Assuming that we have a 5-dimensional one-hot vector and want to turn it into a 2-dimensional word embedding, we define the word embedding matrix as trainable weights of the corresponding size and multiply the two.</p>
  <Latex>{String.raw`
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
  `}</Latex>
  <p>The multiplication will select the correct row and result in a 2-dimensional vector, that can be used as input into our sequence model. This operation will be tracked by the autograd package and those weights will update over the time, optimizing the embedding representation.</p>
  <p>In practice all major frameworks have a dedicated embedding layer, that does this operation via a lookup. This is just a much more efficient representation, but the results should be the same. We will look at what that means when we start implementing language models.</p>
  <div class="separator" />
</Container>


<style>
  .text-center {
    text-align: center;
  }

  .word {
    display: inline-block;
    background: var(--main-color-3);
    padding: 2px 10px;
    margin: 5px;
  }

  .vocabulary {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  .vocabulary-row {
    display: flex;
  }

  .vocabulary-word {
    background: var(--main-color-3);
    width: 120px;
    text-align: center;
    margin-bottom: 1px;
    margin-right: 1px;
    border: 1px solid black;
  }
  .vocabulary-idx {
    background: var(--main-color-3);
    width: 50px;
    text-align: center;
    margin-bottom: 1px;
    border: 1px solid black;
  }

  .index {
    text-align: center;
    background-color: var(--main-color-3);
    width: 25px;
    height: 25px;
    margin-right: 4px;
    margin-bottom: 2px;
    display: inline-block;
    font-size: 20px;
  }

  .hot-index {
    background-color: var(--main-color-1);
    text-align: center;
    width: 25px;
    height: 25px;
    margin-right: 4px;
    margin-bottom: 2px;
    display: inline-block;
    font-size: 20px;
  }

  .cold-index {
    background-color: var(--main-color-2);
    text-align: center;
    width: 25px;
    height: 25px;
    margin-right: 4px;
    margin-bottom: 2px;
    display: inline-block;
    font-size: 20px;
  }

  .embedding-entry {
    background-color: var(--main-color-4);
    text-align: center;
    width: 65px;
    height: 25px;
    margin-right: 4px;
    margin-bottom: 2px;
    display: inline-block;
    font-size: 20px;
  }

  .tokens {
    display: flex;
    justify-content: center;
    align-items: center;
  }

</style>



