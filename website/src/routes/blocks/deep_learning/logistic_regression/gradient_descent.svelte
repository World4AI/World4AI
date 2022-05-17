<script>
  import Container from "$lib/Container.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Gradient Descent</h1>
<div class="separator" />

<Container>
  <p>
    Our goal is to find weights <Latex>{String.raw`\mathbf{w}`}</Latex> and the bias
    <Latex>b</Latex> that minimize the cross-entropy loss in a binary classification
    problem.
  </p>

  <Latex
    >{String.raw`
    \arg\min_{\mathbf{w}, b}H(p, q) = - \sum_i \Big[y^{(i)} \log \sigma(\mathbf{w},b)) + (1 - y^{(i)}) (1 - \log(\sigma(\mathbf{w}, b)) \Big] \\
  `}</Latex
  >
  <p>
    Similar to linear regression, logistic regression relies on gradient
    descent. While the derivative is more complex, the chain rule still allows
    us to relatively easily find the optimum.
  </p>
  <p>
    Ultimately we are interested in the gradient of the loss function <Latex
      >H(p, w)</Latex
    > with respect to the weight vector <Latex
      >{String.raw`\nabla_{\mathbf{x}}`}</Latex
    > and the derivative with respect to the bias <Latex
      >{String.raw`\dfrac{\delta}{\delta b} H(p,x)`}</Latex
    >.
  </p>
  <p>
    Once we have those we can use the same gradient descent procedure that we
    used in linear regression.
  </p>
  <Latex
    >{String.raw`\large \mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w \\`}</Latex
  >
  <Latex
    >{String.raw`\large b_{t+1} \coloneqq b_t - \alpha \dfrac{\delta}{\delta b} \\`}</Latex
  >
  <p>
    That meas that all we have to do is to figure out how we can calculate the
    gradients, the rest of the procedure does not differ from linear regression.
  </p>
  <Latex
    >{String.raw`
    H(p, q) = - \sum_i \Big[y^{(i)} \log a + (1 - y^{(i)}) \log(1 - a)\Big] \\
  `}</Latex
  >
  <div class="separator" />
  <Latex
    >{String.raw`
    \dfrac{\delta}{\delta a}H(p, q) = y^{(i)} \dfrac{1}{a} - (1 - y^{(i)}) \dfrac{1}{1 - a}
  `}</Latex
  >
  <div class="separator" />
  <Latex>{String.raw`a = \sigma(z) = \dfrac{1}{1 + e^{-z}}`}</Latex>
  <div class="separator" />
  <Latex
    >{String.raw`\dfrac{\delta}{\delta z}a = \sigma(z) (1 - \sigma(z)) `}</Latex
  >
  <div class="separator" />
  <Latex>{String.raw`z = \mathbf{w^Tx}+b`}</Latex>
  <div class="separator" />
  <Latex>{String.raw`\dfrac{\delta}{\delta w_j} z = x_j`}</Latex>
  <div class="separator" />
  <Latex>{String.raw`\dfrac{\delta}{\delta b} z = b`}</Latex>
  <div class="separator" />
</Container>
