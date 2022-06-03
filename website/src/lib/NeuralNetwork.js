import * as tf from "@tensorflow/tfjs";
import { writable } from 'svelte/store';

class NeuralNetwork {
  constructor(alpha, sizes, features, labels) { this.alpha = alpha;
    this.sizes = sizes;
    this.features = features;
    this.labels = labels;
    this.numLayers = sizes.length;

    // weights and biases
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.netInputs = [];
    this.dW = [];
    this.db = [];

    //weight initialize
    for (let layer = 1; layer < this.numLayers; layer++) {
      let w = tf.randomNormal([sizes[layer], sizes[layer - 1]]);
      let b = tf.randomNormal([1, sizes[layer]]);
      this.weights.push(w.arraySync());
      this.biases.push(b.arraySync());
    }

    // performance tracking
    this.lossTracker = [];
    this.accuracyTracker = [];

    // create stores 
    this.weightsStore = writable(this.weights);
    this.biasesStore = writable(this.biases);
    this.activationsStore = writable(this.activations);
    this.netInputsStore = writable(this.netInputs);
    this.lossStore = writable(this.lossTracker);
    this.accStore = writable(this.accuracyTracker);
  }

  forward() {
    let activations = []; 
    let netInputs = [];
    let a = this.features;
    let z;
  
    this.weights.forEach((weight, idx) => {
      let w = tf.tensor(weight);
      let b = tf.tensor(this.biases[idx]);
      z = tf.add(tf.dot(a, w.transpose()), b);
      netInputs.push(z.arraySync());
      a = z.sigmoid();
      activations.push(a.arraySync());
    });

    this.netInputs = netInputs;
    this.activations = activations;

    this.netInputsStore.set(netInputs);
    this.activationsStore.set(activations);
  }

  _predict(features) {
    let a = features;
    let z;
    this.weights.forEach((weight, idx) => {
      let w = tf.tensor(weight);
      let b = tf.tensor(this.biases[idx]);
      z = tf.add(tf.dot(a, w.transpose()), b);
      a = z.sigmoid();
    });

    return a.arraySync();
  }

  predict(features) {
    let predictCallback = this._predict.bind(this, features);
    return tf.tidy(predictCallback);
  }
  
  backward() {
    const dW = [];
    const db = [];

    // step 1, dLdw_output
    let y_hat = tf.tensor(this.activations[this.activations.length - 1]);
    let y = tf.tensor(this.labels)

    // dL/dy_hat
    let dLdA = tf.sub(
      tf.mul(tf.sub(1, y), tf.div(1, tf.sub(1, y_hat))),
      tf.mul(y, tf.div(1, y_hat))
    );
    dLdA = tf.div(dLdA, y.shape[0]);

    // dA/dZ
    let dAdZ = tf.mul(y_hat, tf.sub(1, y_hat));

    // dL/dz
    let delta = tf.mul(dLdA, dAdZ);

    // dZdW
    let dZdW = tf.tensor(this.activations[this.activations.length - 2]);

    let dLdW = tf.dot(delta.transpose(), dZdW);
    let dLdB = tf.sum(delta, 0);

    dW.push(dLdW.arraySync());
    db.push(dLdB.arraySync());

    for (let i = this.numLayers - 3; i >= 0; i--) {
      let w = tf.tensor(this.weights[i+1]);
      let dLdAh = tf.dot(delta, w);
      let a = tf.tensor(this.activations[i]);
      let dAhdZ = tf.mul(a, tf.sub(1, a));
      delta = tf.mul(dLdAh, dAhdZ);

      let dZdW = i === 0 ? this.features : this.activations[i - 1];
      let dLdW = tf.dot(delta.transpose(), dZdW);
      let dLdB = tf.sum(delta, 0);
      dW.unshift(dLdW.arraySync());
      db.unshift(dLdB.arraySync());
    }
    this.dW = dW; 
    this.db = db;
  }

  step() {
    let newWeights = [];
    let newBiases = [];
    
    for (let i = 0; i < this.weights.length; i++) {
      let w = tf.tensor(this.weights[i]);
      let b = tf.tensor(this.biases[i]);

      let dw = tf.tensor(this.dW[i]);
      let db = tf.tensor(this.db[i]); 

      dw = tf.mul(this.alpha, dw);
      db = tf.mul(this.alpha, db);

      w = tf.sub(w, dw);
      b = tf.sub(b, db);

      newWeights.push(w.arraySync());
      newBiases.push(b.arraySync());
    }

    this.weights = newWeights; 
    this.biases = newBiases; 

    this.weightsStore.set(newWeights);
    this.biasesStore.set(newBiases);
  }

  performance() {
    let predictions = tf.tensor(this.activations[this.activations.length-1]);
    let y = tf.tensor(this.labels);

    let estimations = predictions.round();
    let diffs = y.equal(estimations).sum();

    let accuracy = tf.div(diffs, estimations.shape[0]);
    let crossEntropy = tf.metrics.binaryCrossentropy(y, predictions);
  
    this.lossTracker.push(crossEntropy.mean().arraySync());
    this.accuracyTracker.push(accuracy.arraySync());

    this.accStore.set(this.accuracyTracker);
    this.lossStore.set(this.lossTracker);
  }

  epoch() {
    // forward 
    let forwardCallback = this.forward.bind(this);
    tf.tidy(forwardCallback);

    // backward
    let backwardCallback = this.backward.bind(this);
    tf.tidy(backwardCallback);

    // take step
    let stepCallback = this.step.bind(this);
    tf.tidy(stepCallback);

    // measure and save performance
    let performanceCallback = this.performance.bind(this);
    tf.tidy(performanceCallback);
  }
}

export {NeuralNetwork}
