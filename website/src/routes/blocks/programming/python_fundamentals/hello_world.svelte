<script>
  import Highlight from "svelte-highlight";
  import python from "svelte-highlight/src/languages/python";
  import "svelte-highlight/src/styles/github-dark.css";
  const helloWorldCode = '>>> print("Hello World!") \nHello World!';
  const singleLineComment = "# This is a single line comment"
  const multiLineComment = '"""\nThis\nis\na\nmultiline\ncomment\n"""'
</script>


<h1>Hello World</h1>
<p><em>How do you output text to the console and how do you put comments in your code?</em></p>
<div class="separator"></div>

<h2>The print() function</h2>
<p>In a programming language a function is piece of code that has to fulfill a certain task, it has to fulfill a function. How exaclty functions work will be covered in a later chapter. In this section we are going to look at a function that is built into Python. The <code>print()</code> function outputs the text that is defined between the brackets into the console.</p>
<Highlight language="{python}" code="{helloWorldCode}" />
<p>A "Hello World!" program like the one above is the customary first step in a programming language. It allows the student to be able to output text. A skill that is essential in all of programming. Go ahead and copy the code, modify it, play with it. This is a great way to test if your setup is working correctly.</p>
<div class="separator"></div>
<h2>Comments</h2>
<p>Being able to comment your code is the second basic skill that you need to posess before you learn the syntax of Python. A comment is a piece of code that is not being interpreted by Python. It's only function is the documentation of the written code. It is a way of communicating the code either to your collegues or yourself in the future.</p>
<p>If we want to comment only a single line of text, we use the hash/pound # symbol before the comment. The line below the comment is not going to be affected by the comment </p> 
<Highlight language="{python}" code="{singleLineComment}" />
<p>Oftentimes you will need to comment out whole blocks of code or to write down a more involved description. For that purpose you can multiline comments. These are initiated with three quotation marks """ and ended with three quotation marks """. Anything between the quotation marks is regarded as a commen multiline comments. These are initiated with three quotation marks """ and ended with three quotation marks """. Anything between the quotation marks is regarded as a comment multiline comments. These are initiated with three quotation marks """ and ended with three quotation marks """. Anything between the quotation marks is regarded as a commen multiline comments. These are initiated with three quotation marks " " " and ended with three quotation marks " " ". Anything between the quotation marks is regarded as a comment.</p>
<Highlight language="{python}" code="{multiLineComment}" />
<div class="separator"></div>


