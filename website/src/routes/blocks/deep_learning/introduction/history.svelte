<script>
  import Question from "$lib/Question.svelte";
  import BiologicalNeuron from "./_history/BiologicalNeuron.svelte";
  import StepFunction from "./_history/StepFunction.svelte";
  import NeuronScaling from "./_history/NeuronScaling.svelte";
  import NeuronScalingAddition from "./_history/NeuronScalingAddition.svelte";
  import NeuronScalingAdditionActivation from "./_history/NeuronScalingAdditionActivation.svelte";
</script>

<h1>The History of Deep Learning</h1>
<Question>What is the history of deep learning?</Question>
<div class="separator" />

<p>
  In this chapter we will attempt to cover the history of deep learning. In all
  likelihood we will fail at covering all the aspects that lead to the
  development of modern deep learning, so consider this chapter merely a
  starting point.
</p>

<p>
  The history of artificial intelligence is characterized by waves of high hopes
  and high investment followed by disillusionment due to unmet promises and the
  drying up of funding. Neural networks (and by extension deep learning) was
  present in all those waves and the so called AI winters, where there was no
  research.
</p>
<div class="separator" />

<h2>First Wave: Birth of Artificial Neural Networks</h2>
<h3>McCulloch-Pitts Neuron 1943</h3>
<p>
  The first artificial neuron was developed by Warren McCulloch and Walter Pitts<sup
    >1</sup
  >. They tried to come up with a mathematical model that would be able to
  simulate a biological neuron. From what was known at a time about the human
  brain, the scientists extracted the following information.
</p>
<p>
  The biological neuron receives inputs in a form of a chemical signal. Each
  signal can vary in strength based on the strength of the connection. Those
  signals are aggregated in the center of the neuron and if the cumulative
  strength of all inputs is larger than some threshold the signal travels over
  the axon (the long tail of the neuron) to the connected neurons. In that case
  we also say that the neuron is active.
</p>

<BiologicalNeuron />
<p>What is important: input, signal strength, sum, activation</p>
<p>Logic gates</p>
<p>Limitations, not training of weights</p>

<!--Input weighing svg -->
<NeuronScaling />

<!-- this animation demonstrates signal addition in McCulloch Pitts -->
<NeuronScalingAddition />

<StepFunction />
<NeuronScalingAdditionActivation />

<div class="separator" />

<h3>Perceptron 1957</h3>
<div class="separator" />

<h3>"Perceptrons" by Minsky and Papert 1969</h3>
<div class="separator" />

<h2>Second Wave: Neural Networks in the Golden Age of Expert Systems</h2>

<h3>Hopfield Network</h3>
<h3>Boltzman Machine</h3>
<h3>Restricted Boltzman Machine</h3>
<h3>Backpropagation 1986</h3>
<p>Rumelhard and Hinton</p>

<h3>Convolutional Neural Networks 1989</h3>
<p>Neocognitron by Kunihiko Fukushima</p>

<h3>Universal Approximation Theorem</h3>
<h3>Vanishing Gradient and LSTM</h3>
<h3>Deep Belief Network</h3>

<h2>Third Wave: Modern Deep Learning</h2>
<h3>ImageNet</h3>
<h3>GPU and AlexNet</h3>
<h3>GAN</h3>
<h3>AlphaGo</h3>
<h3>Transformer</h3>
<h3>Turing Award</h3>

<h2>Notes</h2>
