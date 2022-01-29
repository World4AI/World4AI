<script>
  import Question from "$lib/Question.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
  import Highlight from "$lib/Highlight.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Loops</title>
  <meta
    name="description"
    content="Loops allow us to perform similar actions several times in a row. Python supports while and for loops and also supports continue and break mechanisms."
  />
</svelte:head>

<h1>Loops</h1>
<Question
  >What are loops and what types of loops are available in Python?</Question
>
<div class="separator" />

<p>
  The easiest way to introduce the benefits of loops in a programming language
  is most likely to show the limitations of the programming we have performed so
  far. In the example below we need to execute the function <Highlight
    >say_word()</Highlight
  > five times in a row. And the only way we know how to perform that task is to
  actually write the function name five times.
</p>
<Code
  code={`
def say_word(word):
    print(word)

say_word("Hello")
say_word("World")
say_word("How")
say_word("are")
say_word("you")
say_word("doing")
say_word("?")
`}
/>
<Repl
  code={`
Hello
World
How
are
you
doing
?`}
/>
<p>
  Loops provide us with the ability to perform similar actions several times,
  making the code above cleaner and more scalable. In Python there are two main
  types of loops: <Highlight>for</Highlight> loops and <Highlight
    >while</Highlight
  > loops.
</p>
<div class="separator" />

<h2>for Loops</h2>
<p>
  A <Highlight>for</Highlight> loop is used to iterate over a sequence, like a list
  or a dictionary. Each element in the sequence is provided to the indented block
  below the <Highlight>for</Highlight> statement, starting with the first element
  and ending with the last element in the sequence.
</p>
<p>
  We can implement the same example from above in a much cleaner way using a <Highlight
    >for</Highlight
  > loop.
</p>
<Code
  code={`
words = ["Hello", "World", "How", "are", "you", "doing", "?"]
def say_word(word):
    print(word)

for word in words:
    say_word(word)
`}
/>
<p>
  The variable <Highlight>word</Highlight> in the <Highlight
    >for word in words</Highlight
  > statement takes each of the values in the <Highlight>words</Highlight> list at
  a time. This variable is then provided to the <Highlight>say_word</Highlight> function,
  which prints the correspoinding value in the list. The benefit might not be obvious
  at a first glance, but consider that the list might come from an external source
  and consist of millions of values.
</p>
<div class="separator" />

<h2>while Loops</h2>
<p>
  When you deal with <Highlight>for</Highlight> loops you generally know how many
  iterations your loops is going to take: as many as there are elements in the sequence.
  So you are basically saying to the computer: "Do something for each element in
  the sequence". When we deal with a <Highlight>while</Highlight> loop, we are telling
  to the computer: "Keep going while this condition is true" and we do not necessarily
  know after how many iterations this condition is met.
</p>
<p>
  Computer games for example use a <Highlight>while</Highlight> loop at the core.
  The game implements a so called game loop, that keeps running until the user decides
  to quit the game. We never know in advance how many frames the user is going to
  spend playing the game.
</p>
<p>
  Below is a stylized example of a game loop. At each iteration the <Highlight
    >while</Highlight
  > loop checks if the variable <Highlight>is_running</Highlight> evaluates to <Highlight
    >True</Highlight
  >. If it does, the loop runs for at leas one more iteration. We utilize the <Highlight
    >input</Highlight
  > function to check if the user has input the word "END" into the command line.
  If the user input actually corresponds to "END" we set the value of <Highlight
    >is_running</Highlight
  > to <Highlight>False</Highlight> and the loop is going to be inerrupted the next
  time we check the condition in the head of the loop.
</p>
<Code
  code={`
is_running = True
iteration = 1

while is_running:
    print(f'Iteration Nr. {iteration}')
    user_input = input("Please type the word END if you would like to stop the game: ")
    if user_input == "END":
        is_running = False
    iteration+=1
  `}
/>
<div class="separator" />

<h2>break</h2>
<p>
  The <Highlight>break</Highlight> statement allows us to exit a <Highlight
    >for</Highlight
  > or a <Highlight>while</Highlight> loop at any arbitrary iteration.
</p>
<p>
  Below we start an infinite loop with a <Highlight>while True:</Highlight> statement.
  This type of a loop will continue forever unless we build in a break condition.
  We increment the <Highlight>iteration</Highlight> variable by one until the value
  of 5 is reached. At that point we use <Highlight>break</Highlight> to escape the
  loop.
</p>
<Code
  code={`
iteration = 1

while True:
    print(f'Iteration Nr. {iteration}')

    if iteration == 5:
        break
    iteration+=1
  `}
/>
<Repl
  code={`
Iteration Nr. 1
Iteration Nr. 2
Iteration Nr. 3
Iteration Nr. 4
Iteration Nr. 5
  `}
/>
<div class="separator" />

<h2>continue</h2>
<p>
  The <Highlight>continue</Highlight> statement allows us to skip the current iteration
  and to jump right to the next one.
</p>
<p>
  Below we create the <Highlight>iteration</Highlight> variable. At each iteration
  of the loop we increment the value of the variable by one and print the new value.
  If the <Highlight>iteration</Highlight> variable equals 5 we skip the rest of the
  iteration and continue to the next iteration.
</p>
<Code
  code={`
iteration = 0

while True:
    iteration += 1

    if iteration == 5:
        continue
    print(f'Iteration Nr. {iteration}')

    if iteration == 10:
        break
  `}
/>
<Repl
  code={`
Iteration Nr. 1
Iteration Nr. 2
Iteration Nr. 3
Iteration Nr. 4
Iteration Nr. 6
Iteration Nr. 7
Iteration Nr. 8
Iteration Nr. 9
Iteration Nr. 10
  `}
/>
<div class="separator" />

<h2>range</h2>
<p>
  The <Highlight>for</Highlight> loop is designed to loop through all contents of
  a sequence, like a list or a dictionary. Often in programming we need to loop through
  a sequence of numbers. With <Highlight>range</Highlight> we can create an immutable
  sequence of numbers, that we can utilize in <Highlight>for</Highlight> loops.
</p>
<p>
  <Highlight>range(start, end, step)</Highlight> takes up to 3 arguments as input.
  The <Highlight>start</Highlight> argument determines the first number in the sequence.
  The <Highlight>end</Highlight> parameter determines up to which number the sequence
  will be created. This number is non inclusive, which means that if we set start
  to 0 and end to 10, the numbers that will be created will include 1 to 9, but not
  10. The last parameter, <Highlight>step</Highlight> determines the distance between
  the individual numbers in the sequence. For example <Highlight
    >range(0, 10, 3)</Highlight
  > will generate a sequence that will contain 0, 3, 6 and 9.
</p>
<p>
  Most likely we will require a sequence that starts at 0 and has a step size of
  1. In that case we can utilize the simplified version of range: <Highlight
    >range(end)</Highlight
  >.
</p>
<Code
  code={`
for i in range(5):
    print(i)
    `}
/>
<Repl
  code={`
0
1
2
3
4
  `}
/>
<div class="separator" />

<h2>enumerate</h2>
<p>
  It is not uncommon to loop through a sequence and require the corresponding
  number of the element in the sequence. For that purpose we can combine the <Highlight
    >enumerate</Highlight
  > function with a <Highlight>for</Highlight> loop. In that case at each iteration
  the loop gets a tuple consisting of an index, value pair.
</p>
<Code
  code={`
courses = ['mathematics', 'science', 'english']

for i, course in enumerate(courses):
    print(f'Course Nr. {i} is {course}')
  `}
/>

<Repl
  code={`
Course Nr. 0 is mathematics
Course Nr. 1 is science
Course Nr. 2 is english
  `}
/>
<div class="separator" />

<h2>zip</h2>
<p>
  Sometimes we want to combine two elements that share the same index, but are
  from two different sequences. The <Highlight>zip</Highlight> function is most likely
  the cleanest approach to achieve that goal.
</p>
<Code
  code={`
courses = ['mathematics', 'science', 'english']
teachers = ['Strang', 'Einstein', 'Shakespeare']

for course, teacher in zip(courses, teachers):
    print(f'The {course} course is taught by Prof. {teacher}.')
  `}
/>
<Repl
  code={`
The mathematics course is taught by Prof. Strang.
The science course is taught by Prof. Einstein.
The english course is taught by Prof. Shakespeare.
  `}
/>
<div class="separator" />
