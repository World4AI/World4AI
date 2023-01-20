// Here we create an autodiff package
//based on micrograd by Andrej Karpathy
class Value {
  constructor(data, _children = [], _op = "") {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = _children;
    this._op = _op;
  }

  add(other) {
    other = other instanceof Value ? other : new Value(other);
    let out = new Value(this.data + other.data, [this, other], "+");

    let _backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    out._backward = _backward;

    return out;
  }

  mul(other) {
    other = other instanceof Value ? other : new Value(other);
    let out = new Value(this.data * other.data, [this, other], "*");

    let _backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    return this.add(other.neg());
  }

  div(other) {
    other = other instanceof Value ? other : new Value(other);
    return this.mul(other.pow(-1));
  }

  pow(other) {
    if (!other instanceof Number) {
      throw "A Value object can be only taken to a power of a Number";
    }
    let out = new Value(this.data ** other, [this], `**${other}`);

    let _backward = () => {
      this.grad += (other * this.data) ** (other - 1) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  sigmoid() {
    let out = new Value(1 / (1 + Math.exp(-this.data)), [this], "sigmoid");

    let _backward = () => {
      this.grad += out.data * (1 - out.data) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  relu() {
    let out = new Value(this.data < 0 ? 0 : this.data, [this], "ReLU");

    let _backward = () => {
      this.grad += (this.data > 0) * out.grad;
    };
    out._backward = _backward;
    return out;
  }

  log() {
    let out = new Value(Math.log(this.data), [this], "log");

    let _backward = () => {
      this.grad += (1 / out.data) * out.grad;
    };
    out._backward = _backward;

    return out;
  }

  backward() {
    const topo = [];
    let visited = new Set();

    function buildTopo(v) {
      if (!visited.has(v)) {
        visited.add(v);
        v._prev.forEach((child) => {
          buildTopo(child);
        });
        topo.push(v);
      }
    }
    buildTopo(this);

    this.grad = 1.0;
    topo.reverse();
    topo.forEach((v) => {
      v._backward();
    });
  }
}

class Module {
  zeroGrad() {
    this.parameters().forEach((p) => {
      p.grad = 0;
    });
  }

  parameters() {
    return [];
  }
}

class Neuron extends Module {
  constructor(nin, activation = "sigmoid") {
    super();

    //uniformly initialize weights and bias between -0.5 and 0.5
    let max = 0.5;
    let min = -0.5;
    const init = () => Math.random() * (max - min) + min;

    this.w = [];
    this.b = new Value(init());
    this.activation = activation;

    for (let i = 0; i < nin; i++) {
      let w = new Value(init());
      this.w.push(w);
    }
  }

  forward(x) {
    if (x.length !== this.w.length) {
      throw "Input length does not correspond to weight length";
    }
    let scaledValues = this.w.map((w, i) => w.mul(x[i]));
    let netInput = scaledValues.reduce(
      (accumulator, current) => accumulator.add(current),
      this.b
    );
    if (this.activation === "sigmoid") {
      return netInput.sigmoid();
    } else if (this.activation === "relu") {
      return netInput.relu();
    }
  }

  parameters() {
    return [this.b, ...this.w];
  }
}

class Layer extends Module {
  constructor(nin, nout, activation = "sigmoid") {
    super();
    this.neurons = [];
    for (let i = 0; i < nout; i++) {
      this.neurons.push(new Neuron(nin, activation));
    }
  }

  forward(x) {
    let out = this.neurons.map((neuron) => neuron.forward(x));
    return out.length === 1 ? out[0] : out;
  }

  parameters() {
    return this.neurons.map((neuron) => neuron.parameters()).flat();
  }
}

class MLP extends Module {
  constructor(nin, nouts) {
    super();
    const sz = [nin, ...nouts];
    this.layers = [];
    for (let i = 0; i < sz.length - 1; i++) {
      let activation = i === sz.length - 2 ? "sigmoid" : "relu";
      let layer = new Layer(sz[i], sz[i + 1], activation);
      this.layers.push(layer);
    }
  }

  forward(x) {
    let out = x;
    for (let i = 0; i < this.layers.length; i++) {
      out = this.layers[i].forward(out);
    }
    return out;
  }

  parameters() {
    return this.layers.map((layer) => layer.parameters()).flat();
  }
}

export { Value, Neuron, Layer, MLP };
