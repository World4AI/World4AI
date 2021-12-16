<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Functions Basic Usage</title>
  <meta
    name="description"
    content="To define a function in Python we need to use the def keyword followed by a function name and arguments. The function body is contained within an indented block."
  />
</svelte:head>

<h1>Functions Basic Usage</h1>
<Question>How can we create and use simple functions in Python?</Question>
<div class="separator" />

<h2>Function Definition</h2>
<svg
  width="225px"
  version="1.1"
  viewBox="0 0 225 100"
  xmlns="http://www.w3.org/2000/svg"
>
  <text
    x="7.1523666"
    y="23.386892"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="20px"
    stroke-width=".64457"
    style="line-height:1.25"
    xml:space="preserve"
    ><tspan x="7.1523666" y="23.386892" font-size="20px" stroke-width=".64457"
      >def function_name():</tspan
    ></text
  >
  <text
    x="64.521957"
    y="51.325222"
    fill="var(--text-color)"
    font-family="sans-serif"
    font-size="20px"
    stroke-width=".64457"
    style="line-height:1.25"
    xml:space="preserve"
    ><tspan x="64.521957" y="51.325222" font-size="20px" stroke-width=".64457"
      >function body</tspan
    ></text
  >
  <g fill="none" stroke="var(--text-color)">
    <g stroke-width=".99692">
      <circle cx="68.794" cy="61.526" r="2.4923" />
      <circle cx="68.794" cy="76.468" r="2.4923" />
      <circle cx="68.794" cy="91.735" r="2.4923" />
    </g>
    <g>
      <path d="m7.4343 45h19.337" stroke-width=".59676px" />
      <g stroke-width=".64457px">
        <path d="m39.663 45h19.337" />
        <path d="m29.994 38.554v12.891" />
        <path d="m36.44 38.554v12.891" />
      </g>
      <path d="m7.4343 92.554h19.337" stroke-width=".59676px" />
      <g stroke-width=".64457px">
        <path d="m39.663 92.554h19.337" />
        <path d="m29.994 86.109v12.891" />
        <path d="m36.44 86.109v12.891" />
      </g>
    </g>
  </g>
</svg>
<p>
  We can use the <Highlight>def</Highlight> keyword to create a function. The <Highlight
    >def</Highlight
  > keyword is directly followed by a unique function name, which is the unique identifier
  by which we can later invoke (call) the function. The round brackets <Highlight
    >( )</Highlight
  > that come right after the function name are empty, if the function does not have
  any inputs. Those inputs are called function arguments and are going to be covered
  extensively in a separate chapter. Finally the function head is closed by a colon
  <Highlight>:</Highlight>.
</p>
<p>
  The logic of the function is contained within the function body that is
  contained within the indented piece of block. Usually the indentation
  corresponds to 4 white spaces, but 2 are also a common choise. The only
  requirement is to keep the indentation consistent.
</p>
<Code
  code={`def hello_function():
    print('Hello from a function!')`}
/>
<p>
  To call a function we need to write the function name followed by round
  brackets.
</p>
<Code code={"hello_function()"} />
<Repl code={"Hello from a function!"} />

<p>
  Below is an additional Python function example to get a better feel for
  function definitions and calls.
</p>
<Code
  code={`def one_plus_one():
    result = 1 + 1
    print(result)
    
one_plus_one()`}
/>

<Repl code={"2"} />

<div class="separator" />
