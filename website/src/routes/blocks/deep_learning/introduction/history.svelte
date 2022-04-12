<script>
  import Question from "$lib/Question.svelte";
  import BiologicalNeuron from "./_history/BiologicalNeuron.svelte";
  import StepFunction from "./_history/StepFunction.svelte";
  import NeuronScaling from "./_history/NeuronScaling.svelte";
  import NeuronScalingAddition from "./_history/NeuronScalingAddition.svelte";
  import NeuronScalingAdditionActivation from "./_history/NeuronScalingAdditionActivation.svelte";
  import Latex from "$lib/Latex.svelte";
  import Highlight from "$lib/Highlight.svelte";
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
  >. They tried to come up with a mathematical model of the brain that would be
  able to simulate a biological neuron. From what was known at a time about the
  human brain, the scientists extracted the following information.
</p>
<p>
  The biological neuron receives inputs in a form of a chemical signal. Each
  signal can vary in strength based on the strength of the connection. Those
  signals are aggregated in the center of the neuron. If the cumulative strength
  of all inputs is larger than some threshold the signal travels over the axon
  (the long tail of the neuron) to the connected neurons. In that case we also
  say that the neuron is active.
</p>

<BiologicalNeuron />
<p>
  Let us assume for a second that we are in the position of McCulloch and Pitts
  and would like to emulate the above described behaviour. What would be the key
  characteristics of an artificial neuron?
</p>
<p>
  For once the neuron needs the ability to receive inputs and those inputs need
  to somehow vary in strength. As we are dealing with a mathematical simulation,
  the inputs can be only numbers. In the examples below we will assume that all
  the inputs amount to 1 in order to make the illustrations more intuitive. The
  strength of the signal can be changed with the help of a scaling factor,
  called weight. We therefore adjust the strength of the input by multiplying
  the input <Latex>X</Latex> with a corresponding weight <Latex>W</Latex>. If
  the weight is above 1, the input signal is amplified, if the input is between
  0 and 1 the input signal is dampened. Additionally the strength of the signal
  can be reversed by multiplying the signal with a negative number.
</p>
<p>
  Below is a simple interactive example that allows you to vary the weight and
  observe how the output signal changes. The color of the output signal is blue
  when the output signal is positive and red when it becomes negative. The
  stroke width of the signal output depends on the size of the weight.
</p>
<!--Input weighing svg -->
<NeuronScaling />
<p>
  In the next step we need to figure out the behaviour of the neuron when we
  deal with multiple inputs. Similar to the biological neuron, the input signals
  are accumulated. In the the mathematical model of McCulloch and Pitts the
  weighted inputs are summed up.
</p>
<p>
  Multiplying each input <Latex>X_i</Latex> by a corresponding weight <Latex
    >W_i</Latex
  > and building a sum out of these produces is called a <Highlight
    >weighted sum</Highlight
  >. Mathematically we can express this idea as <Highlight
    ><Latex>{String.raw`\sum_i X_iW_i`}</Latex></Highlight
  >.
</p>
<p>
  In the interactive example below, you can vary the weights of two inputs and
  observe how the weighted signals are accumulated.
</p>
<!-- this animation demonstrates signal addition in McCulloch Pitts -->
<NeuronScalingAddition />

<p>
  A final component that we are missing is the ability of being active based on
  the accumulated input strength. For that McCulloch and Pitts used a simple
  step function. If the weighted sum of the inputs is above a threshold <Latex
    >b</Latex
  > (b stands for bias term) the output is 1, else the output is 0. A function that
  takes the weighted sum as input and determines the activation status of a neuron
  is commonly refered to as the <Highlight>activation function</Highlight>.
</p>
<p>
  Below is a step function with a bias of 0. You can move the slider to observe
  how the shape of the step function changes due to a different bias.
</p>
<StepFunction />
<p>
  The last interactive example allows you to vary two weights and a bias term.
  The output is always either 0 or 1.
</p>
<NeuronScalingAdditionActivation />
<p>
  What we discussed so far is the full fledged artificial neuron. The
  representation of the strength of the signal through a weight, the
  accumulation of the input signals through a sum and the ability to declare the
  neuron as active or inactive through an activation function is still at the
  core of modern deep learning. The ideas developed by McCulloch and Pitts stood
  the test of time.
</p>
<div class="separator" />

<h3>Perceptron 1957</h3>
<div class="separator" />

<h3>"Perceptrons" by Minsky and Papert 1969</h3>
<p>Logic gates</p>
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
