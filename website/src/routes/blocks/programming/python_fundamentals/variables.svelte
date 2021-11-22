<script>
  import Question from '$lib/Question.svelte';
  import Pointers from '$lib/programming/fundamentals/variables/Pointers.svelte';
  import GarbageCollection from '$lib/programming/fundamentals/variables/GarbageCollection.svelte';
  import Code from '$lib/Code.svelte';
  const helloWorldCode = '>>> print("Hello World!") \nHello World!';
  const helloWorldWithVariables = '>>> sentence = "Hello World!" \n>>> print(sentence) \nHello World!'
  const helloWorldWithVariables2 = '>>> sentence = "Hello World!" \n>>> print(sentence) \nHello World!\n>>> sentence = "Goodbye World!"\n>>> print(sentence)\nGoodbye World!'
  const pointersCode = ">>> sentence = 'Hello World'\n>>> sentence_2 = sentence\n>>> print(sentence_2)\nHello World\n>>> id(sentence)\n139657570418736\n>>> id(sentence_2)\n139657570418736"
</script>

<svelte:head>
    <title>World4AI | Programming | Variables </title>
    <meta name="description" content="In Python variables are used to access and modify data. Variables are pointers to the memory location of that data.">
</svelte:head>

<h1>Variables</h1>
<Question>What are variables and how are they used in Python?</Question>
<div class="separator"></div>
<h2>What are variables?</h2>
<p>In a programming language like Python we often deal with some data that has to be loaded, manipulated and output multiple times.</p>
<Code code={helloWorldCode} />
<p>In the above example the string "Hello World!" is the data we are working with. The print statement uses the "Hello World!" string once. After that Python discards the two words. Yet this is not how programming usually looks like. In most cases you want to have access to the "Hello World!" string. You want to be able to modify it and to possible print the modified value over and over again. For that purpose variables can be used. A variable allows a programmer to use a descriptive name to access and possibly modify the data.</p> 
<p class="info">Variables allow programs to access and modify the data in the computer memory.</p>
<Code code={helloWorldWithVariables} />
<p>In the example above the variable with the name <code>sentence</code> is created and assigned a value "Hello World!". After that point <code>sentence</code> can be accessed and used from the <code>print</code> statement.</p>
<Code code={helloWorldWithVariables2} />
<p>The term <strong>variable</strong> starts to make a lot of sense once you realize how variables actually work. Variables are named for their ability to change the value that they are assigned to. Simply put, variables are <em>variable</em> (meaning flexible) in what they can contain. In the above example the <code>sentence</code> variable is first assigned the value "Hello World!" and then reassigned a value "Goodbye World!".</p>
<div class="separator"></div>
<h2>Pointers</h2>
<p>When we talk about variables, what we actually talk about is a convenient name, that helps the programmer to find the corresponding data. The data is located somewhere in your computer memory and keeping track of it without a human readable tag would be impracticle. Internally the address is usually reprsented by a hexadecimal number like 0x7fff1275c05c, which is wildly inconvenient. The variables in Python are essentially pointers to the location of the data and not the data itself.</p>
<Code code={pointersCode}/>
<p>The above example shows how pointers work in Python. In the first line a data object is created and the variable <code>sentence</code> points at the address of the memory location of the 'Hello World' string. In the second line an additional variable is created that points to the same memory location, no additonal data needs to be created. When we use the <code>print()</code> function the  same content of the data is output in the console. To show that this is actually the same object we can use the <code>id()</code> function. When we apply that function to a variable this function returns the unique integer id of that object. For sentence 1 and sentence 2 the id is exactly the same.</p>
<div class="flex-center">
    <Pointers />
</div>
<p>The animation exemplifies that behaviour. 'Hello World' is stored somewhere in the computer memory and both variables point at the same memory location.</p>
<p class="info">In Python a variable is a convenient name for a memory location of the data.</p>
<div class="separator"></div>
<h2>Garbage Collection</h2>
<Code code={helloWorldWithVariables2} />
<p>The above code snippet is the same that we used to explain that variables can be reassigned a different piece of data. When we reassign a variable it is possible that no variable points at a memory location. The object with the "Hello World!" text inside has no reference that points at it, yet it occupies some space in the memory of a computer. In other programming languages this "lost" data might constitute a problem, but Python has a so called garbage collector. Once a piece of data with no variable pointing at it is detected, it is automatically removed to clean up the memory. </p>
<div class="flex-center">
    <GarbageCollection />
</div>
<p class="info">A garbage collector cleans up the data that has no variable referencing it.</p>
<div class="separator"></div>
<h2>Variable Names</h2>
<p>How you name your variables is ultimately up to you, but there is a "Pythonic" way to name variables. The variable names can contain any letters [a-z], [A-Z], any numbers [0-9] (although a variable name can not start with a number) and an underline _. In most cases it is advisable to use the "snake case" convention in Python. All letters have be lowercase, while individual words are separated by underlines. For example <em>this_is_a_variable</em> is a Pythonic way to name a variable, although you should probably select a more descriptive name.</p>
<p>Additionally there are some reserved words in Python that can not be used as variable names. These words are core elements of the language and you will encounter them as we progress though this block of lectures. If you are curious you can input <code>help('keywords')</code> into the prompt or your text editor to display the reserved words.</p>
<p class="info">Use camel case when naming the variables.</p>
<div class="separator"></div>
