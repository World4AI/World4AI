<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Conditionals</title>
  <meta
    name="description"
    content="Conditionals in Python allow us to create branching code, where a certain part of a code is only executed, when a particular condition is met. For that purpose Python supports if, elif and else statements."
  />
</svelte:head>

<h1>Conditionals</h1>
<Question
  >What are conditionals in programming and how can we use them in Python?</Question
>
<div class="separator" />
<svg version="1.1" viewBox="0 0 570 320" xmlns="http://www.w3.org/2000/svg">
  <g stroke="var(--text-color)">
    <g fill="none" stroke-width="1px">
      <path d="m320.91 26.775-144.69 86.523" />
      <path d="m320.91 26.775 144.69 86.523" />
      <path d="m176.22 113.3-75.761 86.523" />
      <path d="m176.22 113.3 75.761 86.523" />
      <path d="m465.6 113.3-75.761 86.523" />
      <path d="m465.6 113.3 75.761 86.523" />
      <path d="m100.46 199.82-75.761 86.523" />
      <path d="m100.46 199.82 75.761 86.523" />
    </g>
    <g fill="var(--text-color)" stroke="black">
      <circle cx="320.91" cy="26.775" r="20" />
      <circle cx="176.22" cy="113.3" r="20" />
      <circle cx="465.6" cy="113.3" r="20" />
      <circle cx="100.46" cy="199.82" r="20" />
      <circle cx="251.99" cy="199.82" r="20" />
      <circle cx="389.84" cy="199.82" r="20" />
      <circle cx="541.36" cy="199.82" r="20" />
      <circle cx="24.703" cy="286.34" r="20" />
      <circle cx="176.22" cy="286.34" r="20" />
    </g>
  </g>
</svg>
<p>
  In programming we often require a certain part of the code to be executed only
  if a certain condition is met. In other words we need to be able to write
  programs that support branching. Different branches of the code tree need to
  be accessed at different conditions.
</p>
<p>
  Conditionals are the programming tools that are used for that purpose.
  Conditionals evaluate boolean variables and allow or disallow the program to
  access certain part of the code based on the value of a the boolean. Python
  provides the following conditionals: <Highlight>if</Highlight>, <Highlight
    >elif</Highlight
  > and <Highlight>else</Highlight> statements.
</p>
<div class="separator" />

<h2>If Statement</h2>
<p>
  The <Highlight>if</Highlight> statement tests weather the boolean value that is
  followed by the statement corresponds to <Highlight>True</Highlight>. If it
  does, the indented block below is executed, otherwise the block is ignored.
</p>
<p>
  In the below example we create two boolean variables. Only the variable <Highlight
    >left</Highlight
  > evaluates to <Highlight>True</Highlight> which leads to the corresponding print
  statement.
</p>
<Code
  code={`
left = True
right = False

if left:
    print("We can go left")

if right:
    print("We can go right")
`}
/>
<Repl
  code={`
We can go left
  `}
/>
<p>
  The if statement can evaluate arbitrarily comlex booleans expressions. As long
  as the expression to the right of the <Highlight>if</Highlight> statement evaluates
  to
  <Highlight>True</Highlight> the indented code will be executed.
</p>
<Code
  code={`
weather = "warm"
rain_probability = 0.2
is_friday = False

if weather == "warm" and rain_probability < 0.4 or is_friday:
    print("We should go outside")
  `}
/>
<Repl
  code={`
We should go outside
  `}
/>
<div class="separator" />
<h2>Else Statement</h2>
<p>
  The <Highlight>else</Highlight> statement has to follow the <Highlight
    >if</Highlight
  > statement. The indented block below the <Highlight>else</Highlight> statement
  will only be executed, if the <Highlight>if</Highlight> statement above was not
  executed. Only one branch of the code can be executed.
</p>
<Code
  code={`
weather = "cold"
rain_probability = 0.8
is_friday = False

if weather == "warm" and rain_probability < 0.4 or is_friday:
    print("We should go outside")
else:
    print("We should stay inside")
    `}
/>
<Repl
  code={`
We should stay inside
  `}
/>
<div class="separator" />

<h2>Elif Statement</h2>
<p>
  There is often a need to combine an <Highlight>else</Highlight> statement with
  an <Highlight>if</Highlight> statement. In Python this is done with an <Highlight
    >elif</Highlight
  > statement. The statement is only evaluated if the <Highlight>if</Highlight> and
  all
  <Highlight>elif</Highlight> staments above were evaluated to <Highlight
    >False</Highlight
  >.
</p>
<Code
  code={`
weather = "cold"
rain_probability = 0.8
is_friday = False
is_hungry = True

if weather == "warm" and rain_probability < 0.4 or is_friday:
    print("We should go outside")
elif is_hungry: 
    print("We should go to a restaurant")
else:
    print("We should stay inside")

  `}
/>
<Repl
  code={`
We should go to a restaurant
  `}
/>
<div class="separator" />

<style>
  svg {
    max-width: 450px;
  }
</style>
