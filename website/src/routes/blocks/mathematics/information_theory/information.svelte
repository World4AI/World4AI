<script>
  import Question from "$lib/Question.svelte";
  import Latex from "$lib/Latex.svelte";
</script>

<h1>Information</h1>
<Question>How is information defined?</Question>
<div class="separator" />

<p>
  We measure information using specific information units. The most common unit
  of information is the so called bit, which is short for "binary unit".
</p>
<p>
  Information is closely related to probability. We expect unlikely events to
  provide more information than very likely events. In fact 1 bit of information
  reduces uncertainty by exactly 2 and 2 bits of information reduce uncertainty
  by exactly 4.
</p>
<p>
  If we take a fair coin as an example, then the message that the toss produced
  the value of "HEAD" for example, reduces the uncertainty by 50% and therefore
  contains exactly one bit of information.
</p>
<p>
  We can generalize this idea using <Latex
    >{String.raw`(\frac{1}{2})^I = p`}</Latex
  >, where <Latex>p</Latex> is a probability of an event and <Latex>I</Latex> is
  the information in bits. If the probability is 50%, the information content is
  exactly 1 bit. If the probability of an event is 25%, the uncertainty is divided
  by 4 when this event occurs and the information content is 2 bits. Generally speaking
  information increases when the probability decreases: information is inversely
  related to probability. As shown below the corresponding information <Latex
    >I</Latex
  > for a event with a probability <Latex>p</Latex> is defined as <Latex
    >{String.raw`I = -\log_2(p)`}</Latex
  >
</p>
<Latex
  >{String.raw`
\begin{aligned}
& \Big(\frac{1}{2}\Big)^I = p \\
& 2^I = \frac{1}{p} \\
I &= \log_2\Big(\frac{1}{p}\Big) \\
I &= \log_2(1) - \log_2(p) \\
I &= -\log_2(p)
\end{aligned}
  `}</Latex
>

<p>
  Using a logarithm to represent information, instead of using probabilities
  additionally has the benefit of being additive. That means that the
  information content of two independent events <Latex>x_1</Latex>
  and <Latex>x_2</Latex> is just <Latex>I(x_1) + I(x_2)</Latex>.
</p>

<p>
  In machine learning we often use nats for convenience (natural units), which
  use the base <Latex>e</Latex> instead of bits with the binary base. This does not
  pose a problem, as we can simply convert from bits to nats by changing the base
  from <Latex>2</Latex> to <Latex>e</Latex>.
</p>
<div class="separator" />
