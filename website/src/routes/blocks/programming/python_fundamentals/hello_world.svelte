<script>
  import Question from '$lib/Question.svelte';
  import Code from '$lib/Code.svelte';
  const helloWorldCode = '>>> print("Hello World!") \nHello World!';
  const singleLineComment = "# This is a single line comment"
  const multiLineComment = '"""\nThis\nis\na\nmultiline\ncomment\n"""'
  const helpCode = `
Python 3.9.7 (default, Sep 16 2021, 13:09:58)
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
      const printHelpCode = `Help on built-in function print in module builtins:

print(...)
    print(value, ..., sep=' ', end='\\n', file=sys.stdout, flush=False)

    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file:  a file-like object (stream); defaults to the current sys.stdout.
    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.
(END)
   `
</script>

<svelte:head>
    <title>World4AI | Programming | Hello World</title>
    <meta name="description" content="To create a 'Hello World' program in Python we need to use the pring() function.">
</svelte:head>

<h1>Hello World</h1>
<Question>How do you output text to the console and how do you put comments in your code?</Question>
<div class="separator"></div>

<h2>The print() function</h2>
<p>In a programming language a function is piece of code that has to fulfill a certain task, it has to fulfill a function. How exaclty functions work will be covered in a later chapter. In this section we are going to look at a function that is built into Python. The <code>print()</code> function outputs the text that is defined between the brackets into the console.</p>
<Code code={helloWorldCode} />
<p>A "Hello World!" program like the one above is the customary first step in a programming language. It allows the student to be able to output text. A skill that is essential in all of programming. Go ahead and copy the code, modify it, play with it. This is a great way to test if your setup is working correctly.</p>
<div class="separator"></div>
<h2>Comments</h2>
<p>Being able to comment your code is the second basic skill that you need to posess before you learn the syntax of Python. A comment is a piece of code that is not being interpreted by Python. It's only function is the documentation of the written code. It is a way of communicating the code either to your collegues or yourself in the future.</p>
<p>If we want to comment only a single line of text, we use the hash/pound # symbol before the comment. The line below the comment is not going to be affected by the comment </p> 
<Code code={singleLineComment} />
<p>Oftentimes you will need to comment out whole blocks of code or to write down a more involved description. For that purpose you can multiline comments. These are initiated with three quotation marks """ and ended with three quotation marks """. Anything between the quotation marks is regarded as a commen multiline comments. These are initiated with three quotation marks """ and ended with three quotation marks """. Anything between the quotation marks is regarded as a comment multiline comments. These are initiated with three quotation marks """ and ended with three quotation marks """. Anything between the quotation marks is regarded as a commen multiline comments. These are initiated with three quotation marks " " " and ended with three quotation marks " " ". Anything between the quotation marks is regarded as a comment.</p>
<Code code={multiLineComment} />
<div class="separator"></div>
<h2>The help() function</h2>
<p>Python has a built-in help functionality. If you want to start the interactive session, type <code>help()</code> while using the Python interpreter.</p>
<Code code={helpCode} />
<p>Once the interactive session starts we can type the name of functions, modules etc to receive helpful information. If we type for example <code>print</code> in the command line, the interactive session returns the following text.</p>
<Code code={printHelpCode} />
<p>We will make use of the <code>help()</code> functionality throughout the lectures. Many custom libraries write documentations that <code>help()</code> can acess and we will learn how to do the same, once we start implementing our own functions and modules.</p>
<div class="separator"></div>

