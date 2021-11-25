<script>
 import {fade} from 'svelte/transition';
 import Question from '$lib/Question.svelte';
 import Code from '$lib/Code.svelte';
</script>

<svelte:head>
    <title>World4AI | Programming | Gettin Help for Python</title>
    <meta name="description" content="It is essential to know how to get help for Python, when you are stuck. Python offers an integrated help() function and the official Python website offers a wealth of knowledge.">
</svelte:head>

<h1>Getting Help for Python</h1>
<Question>Where can you get help and how can you improve your Python skills?</Question>
<div class="separator"></div>

<div class="flex-center">
 <svg width="200" height="200" version="1.1" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="var(--text-color)" stroke-width=1>
    <path in:fade="{{duration: 1000}}" d="m86.872 143h17.249v21.583h-17.249zm16.74-12.491h-16.23v-13.086q0-8.5822 2.3792-14.105 2.3792-5.5232 10.027-12.831l7.6475-7.5626q4.8434-4.5036 6.9678-8.4973 2.2093-3.9937 2.2093-8.1574 0-7.5626-5.6082-12.236-5.5232-4.6735-14.7-4.6735-6.7128 0-14.36 2.974-7.5626 2.974-15.805 8.6672v-15.975q7.9874-4.8434 16.145-7.2227 8.2423-2.3792 16.995-2.3792 15.635 0 25.067 8.2423 9.5169 8.2423 9.5169 21.753 0 6.4579-3.059 12.321-3.059 5.7781-10.707 13.086l-7.4776 7.3076q-3.9937 3.9937-5.6932 6.288-1.6145 2.2093-2.2943 4.3336-0.50984 1.7844-0.76475 4.3336-0.25492 2.5492-0.25492 6.9678z"/>
    <path in:fade="{{duration: 1000}}" d="m100 5c-52.455 0-95 42.545-95 95 0 52.455 42.545 95 95 95 52.455 0 95-42.545 95-95 0-52.455-42.545-95-95-95zm0 2c51.374 0 93 41.626 93 93 0 51.374-41.626 93-93 93-51.374 0-93-41.626-93-93 0-51.374 41.626-93 93-93z"  stroke-linecap="round" stroke-linejoin="round" style="-inkscape-stroke:none"/>
  </g>
 </svg>
</div>
<p>Before we start our journey through Python it is important to discuss how you can get help when you get stuck. Most people would turn to their search engine to figure out the solution to their problem. While this this is a valid choice, the Python ecosystem provides a wealth of information.</p>
<div class="separator"></div>
<h2>Interactive Help</h2>
<p>Python has a built-in help functionality. If you want to start the interactive session, type <code>help()</code> in the Python command line.</p>
<Code code={
`Python 3.9.7 (default, Sep 16 2021, 13:09:58)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> help()

Welcome to Python 3.9's help utility!

If this is your first time using Python, you should definitely check out
the tutorial on the Internet at https://docs.python.org/3.9/tutorial/.

Enter the name of any module, keyword, or topic to get help on writing
Python programs and using Python modules.  To quit this help utility and
return to the interpreter, just type "quit".

To get a list of available modules, keywords, symbols, or topics, type
"modules", "keywords", "symbols", or "topics".  Each module also comes
with a one-line summary of what it does; to list the modules whose name
or summary contain a given string such as "spam", type "modules spam".

help>`
} />

<p>Once the interactive session starts we can type the name of functions, modules etc to receive helpful information. If we type for example <code>print</code> in the command line, the interactive session returns the following text.</p>
<Code code={
`Help on built-in function print in module builtins:

print(...)
    print(value, ..., sep=' ', end='\\n', file=sys.stdout, flush=False)

    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file:  a file-like object (stream); defaults to the current sys.stdout.
    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.
(END)`
} />
<p></p>
<div class="separator"></div>
<h2>Python Documentation</h2>
<p>The official Python <a target="_blank" href="https://docs.python.org/3/">website</a> provides an extensive tutorial, a library reference, a language reference and book reccomendations. In most cases this is all you need to learn Python. Search engines unfortuntatly do not favour the official documentation, but this is the most up to date and the most comprehensive Python resource.</p>
<div class="separator"></div>
