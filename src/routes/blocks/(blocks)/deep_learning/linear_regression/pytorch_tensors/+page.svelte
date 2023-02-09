<script>
  import Container from "$lib/Container.svelte";
  import PythonCode from "$lib/PythonCode.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<svelte:head>
  <title>PyTorch Tensors - World4AI</title>
  <meta
    name="description"
    content="All deep learning libraries are build around a tensor object. A tensor is primarily used to store matrices of different dimensions and to apply rules of linear algebra."
  />
</svelte:head>

<h1>PyTorch Tensors</h1>
<div class="separator" />

<Container>
  <p>
    Literally all modern deep learning libraries are based on a fundamental
    mathematical object called <Highlight>tensor</Highlight>. We will use this
    object throughout all remaining chapters of this block, no matter if we
    implement something as trivial as linear regresssion or a state of the art
    deep learning architecture.
    <PythonCode code={`import torch`} />
  </p>
  <p>
    According to the PyTorch documentation, <code>torch.Tensor</code> is a
    multi-dimensional matrix containing elements of a single data type. The
    method <code>torch.tensor()</code> is the most straightforward way to create
    a tensor. Below for example we create a tensor object with 2 rows and 3 columns.
  </p>
  <PythonCode
    code={String.raw`tensor = torch.tensor([[0, 1, 2], [3, 4, 5]]) 
print(tensor)`}
  />
  <pre class="text-sm">
tensor([[0, 1, 2],
        [3, 4, 5]])
  </pre>
  <p>
    The method has some arguments, that allow us to control the properties of
    the tensor: <code
      >torch.tensor(data, dtype=None, device=None, requires_grad=False)</code
    >.
  </p>
  <p>
    The <code>data</code> argument is the only required parameter. With this argument
    we provide an arraylike structure, like a list, a tuple or a NumPy ndarray, to
    construct a tensor.
  </p>
  <PythonCode
    code={String.raw`tensor = torch.tensor(data=[[0, 1, 2], [3, 4, 5]])`}
  />
  <p>
    The <code>dtype</code> argument determines the type of the tensor. This
    essentially means, that we have to think about in advance what type of data
    a tensor is supposed to contain. If we do not specify the type explicitly,
    <code>dtype</code>
    is going to be <code>torch.int64</code>, if all of inputs are integers and
    it is going to be <code>torch.float32</code> if even one of the inputs is a
    float. Most neural network weights and biases are going to be
    <code>torch.float32</code>, so for the time being those two datatypes are
    actually sufficient to get us started. When the need arises, we will cover
    more datatypes.
  </p>
  <PythonCode
    code={String.raw`tensor = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32) 
print(tensor.dtype)`}
  />
  <pre class="text-sm">
torch.float32
  </pre>
  <p>
    Tensors can live on different devices, like the cpu, the gpu or tpu and the <code
      >device</code
    >
    argument allows us to create a tensor on a particular device. If we do not specify
    a device, we will use the cpu as the default. For the most part we will be interested
    in moving a tensor to the gpu to get better parallelisation. For that we need
    to have an Nvidia graphics card. We can test if we have a valid graphics card,
    by running
    <code>torch.cuda.is_available()</code>. If the method returns
    <code>True</code>, we are good to go.
  </p>
  <PythonCode
    code={String.raw`# cuda:0 represents the first nvidia device
# theoretically you could have several graphics cards
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tensor = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)`}
  />
  <p>
    The last argument, <code>requires_grad</code> determines whether the tensor needs
    to be included in gradient descent calculations. This will be covered in more
    detail in future tutorials.
  </p>
  <p>
    There are many more methods to create a Tensor. The method <code
      >torch.from_numpy()</code
    >
    turns a numpy ndarray into a PyTorch tensor, <code>torch.zeros()</code>
    returns a Tensor with all zeros and <code>torch.ones()</code> returns a Tensor
    with all ones. We will see more of those methods as we go along. It makes no
    sense to cover all of them without any context.
  </p>
  <p>
    If we need to change the parameters of an already initialized Tensor, we can
    do the adjustments in a later step, primarily using the <code>to</code>
    method of the Tensor class. The <code>to</code> method does not overwrite the
    original Tensor, but returns an adjusted one.
  </p>
  <PythonCode
    code={String.raw`tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])
print(f'Original Tensor: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
tensor = tensor.to(torch.float32)
print(f'Adjusted dtype: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
tensor = tensor.to(device)
print(f'Adjusted device: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
tensor.requires_grad = True
print(f'Adjusted requres_grad: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
`}
  />
  <pre class="text-sm">
Original Tensor: dtype=torch.int64, device=cpu, requires_grad=False
Adjusted dtype: dtype=torch.float32, device=cpu, requires_grad=False
Adjusted device: dtype=torch.float32, device=cuda:0, requires_grad=False
Adjusted requres_grad: dtype=torch.float32, device=cuda:0, requires_grad=True
  </pre>
  <p>
    In practice we are often interested in the shape of a particular tensor. We
    can use use my_<code>tensor.size()</code> or <code>my_tensor.shape</code> to
    find out the dimensions of the tensor.
  </p>
  <PythonCode
    code={String.raw`print(tensor.size())
print(tensor.shape)
`}
  />
  <pre class="text-sm">
torch.Size([2, 3])
torch.Size([2, 3])
  </pre>
  <p>
    PyTorch, like other frameworks that work with tensors, is extremely
    efficient when it comes to matrix operations. These operations are done in
    parallel and can be transfered to the GPU if you have a cuda compatibale
    graphics card. Essentially all of deep learning is based on matrix
    operations, so let"s spend some time to learn how we can invoke matrix
    operations using <code>Tensor</code> objects.
  </p>
  <p>
    We will use two tensors, <Latex>{String.raw`\mathbf{A}`}</Latex> and <Latex
      >{String.raw`\mathbf{B}`}</Latex
    > to demonstrate basic mathematical operations.
  </p>
  <PythonCode
    code={String.raw`A = torch.ones(size=(2, 2), dtype=torch.float32)
B = torch.tensor([[1, 2],[3, 4]], dtype=torch.float32)
`}
  />
  <p>
    We can add, subtract, multiply and divide those matrices using basic
    mathematic operators like <code>+</code>, <code>-</code>, <code>*</code>,
    <code>/</code>. All those operations work elementwise, so when you multiply
    two matrices you won't actually use matrix multiplication that involves dot
    products but elementwise multiplication.
  </p>
  <PythonCode
    code={String.raw`print(A + B)
print(A - B)
print(A * B)
print(A / B)
`}
  />

  <pre class="text-sm">
  tensor([[2., 3.],
          [4., 5.]])
  tensor([[ 0., -1.],
          [-2., -3.]])
  tensor([[1., 2.],
          [3., 4.]])
  tensor([[1.0000, 0.5000],
          [0.3333, 0.2500]])
  </pre>
  <p>
    We can achieve the same results using the explicit methods: <code
      >Tensor.add()<code
        >, <code>Tensor.subtract()</code>, <code>Tensor.multiply()</code>,
        <code>Tensor.divide(). </code></code
      ></code
    >
  </p>
  <PythonCode
    code={String.raw`print(A.add(B))
print(A.subtract(B))
print(A.multiply(B))
print(A.divide(B))
`}
  />
  <p>
    While the above methods do not change the original tensors, each of the
    methods has a corresponding method that changes the tensor in place. These
    methods always end with a <code>_</code>: <code>add_()</code>,
    <code>subtract_()</code>, <code>multiply_()</code>, <code>divide_()</code>.
  </p>
  <PythonCode
    code={String.raw`test = torch.tensor([[1, 2], [4, 4]], dtype=torch.float32)
test.add_(A)
# the test tensor was changed
print(test)
`}
  />
  <pre class="text-sm">
tensor([[2., 3.],
        [5., 5.]])
  </pre>
  <p>
    Probaly one of the most important matrix operations in all of deep learning
    is product of two matrices, <Latex>{String.raw`\mathbf{A \cdot B}`}</Latex>.
    For that purpose we can use the <code>matmul</code> method.
  </p>
  <PythonCode
    code={String.raw`# Equivalent to torch.matmul(A, B)
A.matmul(B)
`}
  />
  <pre class="text-sm">
tensor([[4., 6.],
        [4., 6.]])
  </pre>
  <p>
    Alternatively we can use <code>@</code> as a convenient way to use matrix
    multiplication. This is essentially just a shorthand notation for
    <code>torch.matmul</code>.
  </p>
  <PythonCode
    code={String.raw`# Equivalent to torch.matmul(A, B)
A @ B
`}
  />
  <p>
    A final concept that we would like to mention is the concept of dimensions
    in PyTorch. Often we would like to calculate some summary statistics (like a
    sum or a mean) for a Tensor object. But we would like those to be calculated
    for a particular dimension. We can explicitly set the dimension by defining
    the <code>dim</code> parameter.
  </p>
  <PythonCode
    code={String.raw`t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
`}
  />
  <pre>
tensor(21)
tensor([5, 7, 9])
tensor([ 6, 15])
  </pre>
  <p>
    The very first sum that we calculate in the example below, does not take any
    dimensions into consideration and just calculates the sum over the whole
    tensor. In the second example we calculate the sum over the 0th, the row,
    dimension. That means that for each of the available columns we calculate
    the sum by moving down the rows. When we calculate the sum for the 1st, the
    column dimension, we go over each row and calculate the sum by moving
    through the columns.
  </p>
  <div class="separator" />
</Container>
