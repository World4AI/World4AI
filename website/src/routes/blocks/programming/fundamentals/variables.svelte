<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  import Repl from '$lib/Repl.svelte';
  import PyObject from '$lib/PyObject.svelte';
</script>

<svelte:head>
    <title>World4AI | Programming | Variables </title>
    <meta name="description" content="In Python variables are used to access and modify data. Variables are names that allow you to work with objects in memory.">
</svelte:head>

<h1>Variables</h1>
<Question>What are variables and how are they used in Python?</Question>
<div class="separator"></div>
<h2>What are variables?</h2>
<p>In a programming language like Python we often deal with some data that has to be loaded, manipulated and generally accessed multiple times.</p>
<Code code={'print("Hello World!")'} />
<Repl code={'Hello World!'} />
<p>In the above example the string "Hello World!" is the data we are working with. Python creates a PyObject that holds "Hello World!", prints the text and discards the object right after. Yet this is not how programming usually looks like. In most cases you want to have access to the "Hello World!" string. You want to be able to modify it and to possibly access the modified value over and over again. For that purpose variables can be used. A variable allows a programmer to use a descriptive name to access and possibly modify the data.</p> 
<p class="info">Variables are names that allow programs to access and modify the data in the computer memory.</p>
<Code code={`number = 42
print(number)
`} />                                                                                                          
<Repl code={'42'} />
<p>In the example above we create a variable named "number" with a value of 42. In the next step the variable is used to print the value in the terminal.</p>
<PyObject variable={'number'} />
<p>The variable "number" is used to access the PyObject with the value of 42. Therefore when we talk about variables, what we actually talk about is a convenient name, that helps the programmer to find the corresponding data. The data is located somewhere in your computer memory and keeping track of it without a human readable tag would be impracticle. Internally the address is usually reprsented by a hexadecimal number like 0x7fff1275c05c, which is wildly inconvenient. The variables in Python are like pointers to the location of the data and not the data itself.</p>
<Code code={`number = 42 
number_2 = number
print(number_2)
print(id(number))
print(id(number_2))`}/>
<Repl code={`
42
139657570418736
139657570418736
`} />
<p>The above example shows how variables work in Python. In the first line an object is created and the variable <code>number</code> points at the address of the memory location of the number 42. In the second line an additional variable is created that points to the same memory location, therefore no additonal PyObject needs to be created. When we use the <code>print()</code> function the  same content is output in the terminal. To show that this is actually the same object we can use the <code>id()</code> function. When we apply that function to a variable this function returns the unique integer id of that object. For number and number_2 the id is exactly the same, therefore we are dealing with the same object.</p>
<PyObject variable={'number'} variable2={'number_2'} />
<p>The diagram exemplifies that behaviour. The nuber 42 is stored somewhere in the computer memory and both variables point at the same memory object.</p>
<p class="info">In Python a variable is a convenient name for a memory location of the data.</p>
<Code code={`sentence = "Hello World!"
print(sentence) 
sentence = "Goodbye World!"
print(sentence)
`} />
<Repl code={`Hello World!
Goodbye World!`} />
<p>The term <strong>variable</strong> starts to make a lot of sense once you realize how variables actually work. Variables are named for their ability to be assigned different values (PyObjects). Simply put, variables are <em>variable</em> (meaning flexible) in what they can contain. In the above example the <code>sentence</code> variable is first assigned the value "Hello World!" and then reassigned a value "Goodbye World!".</p>
<div class="separator"></div>
<h2>Variable Names</h2>
<p>How you name your variables is ultimately up to you, but there is a "Pythonic" way to name variables. The variable names can contain any letters [a-z], [A-Z], any numbers [0-9] (although a variable name can not start with a number) and an underline _. In most cases it is advisable to use the "snake case" convention in Python. All letters have be lowercase, while individual words are separated by underlines. For example <em>this_is_a_variable</em> is a Pythonic way to name a variable, although you should probably select a more descriptive name.</p>
<p>Additionally there are some reserved words in Python that can not be used as variable names. These words are core elements of the language and you will encounter them as we progress though this block of lectures. If you are curious you can input <code>help('keywords')</code> into the prompt or your text editor to display the reserved words.</p>
<p class="info">Use camel case when naming the variables.</p>
<div class="separator"></div>
