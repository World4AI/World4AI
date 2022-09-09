<script>
  import Container from "$lib/Container.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Latex from "$lib/Latex.svelte";
  import Table from "$lib/Table.svelte";

  let header = ["Distance to City Centre", "Size", "Price"];
  let data = [
    ["20", "100", 2_000_000],
    ["30", "30", 800_000],
    ["40", "40", 700_000],
    ["30", "60", 1_700_000],
  ];

  let references = [
    {
      author: "Raschka S., Liu Y. Mirjalili V.",
      title: "Machine Learning with PyTorch and Scikit-Learn",
      journal: "",
      year: "2022",
      pages: "9-11",
      volume: "",
      issue: "",
    },
  ];
</script>

<svelte:head>
  <title>World4AI | Deep Learning | Mathematical Notation</title>
  <meta
    name="description"
    content="Mathematical notation in machine learning can be confusing especially if not kept consistent. Here we introduce notation, that is going to be kept consistent throughout the deep learning block."
  />
</svelte:head>

<h1>Mathematical Notation</h1>
<div class="separator" />

<Container>
  <p>
    Under the hood machine learning is based on math. The math is relatively
    simple and you are not required to know more than basic linear algebra,
    calculus and probability theory. Skills that you can improve as you progress
    in this block.
  </p>

  <p>
    The complexity and confusion often arises not from the math itself, but from
    mathematical notation. Unfortunately this notation is not always consistent
    from paper to paper, from book to book and from author to author, so you
    really need to make sure you understand the intentions of the authors.
  </p>
  <p>
    Generally speaking we will utilze the notation that reduces the friction
    between math and computational implementation, in other words once you
    understand the math you should be able to understand the code automatically
    and vice versa.
  </p>
  <p>
    We are not going to completely reinvent the wheel, but instead utilize
    already established notation in books and courses. Mostly we will use the
    notation used by Sebastian Raschka in his books<InternalLink
      type={"reference"}
      id={1}
    /> and courses. The book by Sebastian Raschka is in our self-study recommendation
    list and should thereby create some familiarity.
  </p>
  <p>
    To clarify the mathematical notation we are going to use the example of a
    dataset that contains the features of a house and the corresponding target
    (price).
  </p>
  <Table {data} {header} />
  <p>
    We start with the representation of the feature dataset
    <Latex>{String.raw`\mathbf{X} \in \mathbb{R}^{n\times m}`}</Latex>
    . The dataset is an <Latex>n\times m</Latex> Matrix with <Latex>n</Latex> number
    of samples (rows) and <Latex>m</Latex>
    number of features (columns).
  </p>
  <Latex
    >{String.raw`
  \mathbf{X} = 
  \begin{bmatrix}
  x^{(1)}_1 & x^{(1)}_2 & x^{(1)}_3 & \cdots & x^{(1)}_m \\
  x^{(2)}_1 & x^{(2)}_2 & x^{(2)}_3 & \cdots & x^{(2)}_m \\
  x^{(3)}_1 & x^{(3)}_2 & x^{(3)}_3 & \cdots & x^{(3)}_m \\
  \vdots & \vdots & \vdots & \cdots & \vdots & \\ 
  x^{(n)}_1 & x^{(n)}_2 & x^{(n)}_3 & \cdots & x^{(n)}_m \\
  \end {bmatrix}
  `}</Latex
  >
  <p>
    The superscript <Latex>(i)</Latex> is used to display individual samples in the
    dataset and the subscript <Latex>j</Latex> is used to display the features in
    the dataset. For example <Latex>{String.raw`x^{(1)}_5`}</Latex> references the
    5th feature in the first sample of the dataset.
  </p>
  <p>
    If we apply this notation to the houseprice dataset we get the <Latex
      >4 \times 2</Latex
    >
    representation for the feature matrix.
  </p>
  <Latex
    >{String.raw`
  \mathbf{X} = 
  \begin{bmatrix}
  20^{(1)}_1 & 100^{(1)}_2 \\
  30^{(2)}_1 & 40^{(2)}_2 \\
  40^{(3)}_1 & 40^{(3)}_2 \\
  30^{(4)}_1 & 60^{(4)}_2 \\
  \end {bmatrix}
  `}</Latex
  >
  <p>
    Often we want to show calculations for one single sample in the dataset. For
    purpose that we use row vectors, represented by bold, lowercase letter <Latex
      >{String.raw`\mathbf{x}`}</Latex
    >.
  </p>
  <Latex
    >{String.raw`
  \mathbf{x} = 
  \begin{bmatrix}
  x_1 & x_2 & x_3 & \cdots & x_m
  \end {bmatrix}
  `}</Latex
  >
  <p>
    The labels are represented by a column vecor <Latex
      >{String.raw`\mathbf{y}`}</Latex
    > of size <Latex>n</Latex>.
  </p>
  <Latex
    >{String.raw`
  \mathbf{y} = 
  \begin{bmatrix}
  y^{(1)}  \\
  y^{(2)}  \\
  y^{(3)}  \\
  \vdots \\
  y^{(n)}  \\
  \end {bmatrix}
  `}</Latex
  >
  <p>
    In the house dataset the price is the label represented by the vector <Latex
      >{String.raw`\mathbf{y}`}</Latex
    > of dimension <Latex>4</Latex>.
  </p>
  <Latex
    >{String.raw`
  \mathbf{y} = 
  \begin{bmatrix}
  2000000^{(1)}  \\
  800000^{(2)}  \\
  700000^{(3)}  \\
  1700000^{(4)}  \\
  \end {bmatrix}
  `}</Latex
  >
  <p>
    The weight vector that is used to scale the input vector <Latex
      >{String.raw`\mathbf{x}`}</Latex
    > is represented by the bold letter <Latex>{String.raw`\mathbf{w}`}</Latex>.
  </p>
  <Latex
    >{String.raw`
  \mathbf{w} = 
  \begin{bmatrix}
    w_1 & w_2 & w_3 & \cdots & w_m
  \end {bmatrix}
  `}</Latex
  >
  <p class="warning">
    We do not expect you to learn the above notation by heart. The intention is
    to have a reference that can be used to return to, if this becomes
    necessary. Nor do we assume the notation to be complete. The notation above
    is the foundation that is fundamental to all machine learning topics. Each
    section will enhance the notation to suit the needs of the section.
  </p>
  <Footer {references} />
</Container>
