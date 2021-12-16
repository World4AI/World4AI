<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Functions Inputs and Outputs</title>
  <meta
    name="description"
    content="In Python functions can receive inputs by providing arguments during the function call and return otputs by using the return keyword."
  />
</svelte:head>

<h1>Inputs and Outputs</h1>
<Question>How can we use inputs and outputs in functions?</Question>
<div class="separator" />

<h2>Return Statement</h2>
<p>
  A function does not only calculate or process values, but can also return
  those values to the position the function was initially called. For that
  purpose the <Highlight>return</Highlight> keyword is used.
</p>
<Code
  code={`def one_plus_one():
    result = 1 + 1
    return result
    
result = one_plus_one()
print(result)`}
/>
<Repl code={"2"} />
<p>
  The returned value can be accessed through a variable at a later stage by
  using the assignment operator <Highlight>=</Highlight>.
</p>
<p>
  Once the <Highlight>return</Highlight> keyword is used within a function, the function
  exits and the code that comes after is ignored.
</p>
<Code
  code={`def one_plus_one():
    print("Before return")
    result = 1 + 1
    return result
    print("After return")

    
result = one_plus_one()
print(result)`}
/>
<Repl
  code={`Before return
2`}
/>
<p>The second print statement in the above example is not reachable.</p>
<div class="separator" />

<h2>Parameters and Arguments</h2>
<p>
  Oftentimes we would like the results of our functions to depend on some
  inputs. We can define the parameters (inputs) of the function by listing them
  between the round brackets and separating them by commas. Within the body of
  the function those arguments are used as placeholders to define the logic of
  the function. Once we call a function we have to input actual objects
  (arguments) which are used in place of parameters.
</p>
<Code
  code={`def sum_two_nums(num_1, num_2):
    return num_1 + num_2

result = sum_two_nums(1, 2)
print(result)`}
/>
<Repl code={"3"} />
<p>
  In the example above num_1 and num_2 are parameters of the function. The
  actual inputs 1 and 2 are called arguments of the function. This definition is
  not always used consistently in the programming and Python literature, but
  usually it is clear from the context what the author means.
</p>
<div class="separator" />

<h2>Positional vs Keyword</h2>
<p>
  There are two general options to pass arguments to Python functions: as
  positional arguments or as keyword arguments. Positional arguments are matched
  with the corresponding parameter positiona by position. The first argument is
  matched with the first paramter and so on.
</p>
<p>
  In the example below for example 1 is matched with num_1 and 2 is matched with
  num_2.
</p>
<Code
  code={`def sum_two_nums(num_1, num_2):
    return num_1 + num_2

result = sum_two_nums(1, 2)
print(result)`}
/>
<Repl code={"3"} />
<p>
  When we use keyword arguments we have to specify which object (argument) is
  matched with which parameter by using the explicit parameter name and the
  assignment operator <Highlight>=</Highlight>. The order by which we pass
  paramters is not important.
</p>
<Code
  code={`def sum_two_nums(num_1, num_2):
    return num_1 + num_2

result = sum_two_nums(num_1=1, num_2=2)
print(result)

result = sum_two_nums(num_2=2, num_1=1)
print(result)`}
/>
<Repl
  code={`3
3`}
/>
<p>
  It is also possible to combine positional and keyword arguments, but as the
  two examples below show all positional arguments have to preced all keyword
  arguments. When a positional argument follows a keyword argument Python throws
  an error.
</p>
<Code
  code={`
def sum_two_nums(num_1, num_2):
    return num_1 + num_2

result = sum_two_nums(1, num_2=2)
print(result)
  `}
/>
<Repl code={"3"} />
<Code
  code={`
def sum_two_nums(num_1, num_2):
    return num_1 + num_2

result = sum_two_nums(num_1=1, 2)
  `}
/>
<Repl code={"SyntaxError: positional argument follows keyword argument"} />
<div class="separator" />

<h2>Optional Arguments</h2>
<p>
  Oftentimes functions work well with default arguments. Some machine learning
  models for example have dozens of theoretical inputs, but require only a
  couple of inputs in order to work. Other arguments are optional and use
  default values when they are not explicitly set. In order to create optinal
  arguments we use the assignment operator <Highlight>=</Highlight> in the function
  head.
</p>
<Code
  code={`
def sum_two_nums(num_1, num_2=2):
    return num_1 + num_2

result = sum_two_nums(1)
print(result)
  `}
/>
<Repl code={"3"} />
<p>
  Once we declare an argument as optinal, all following arguments also have to
  be made optional.
</p>
<Code
  code={`
def sum_two_nums(num_1=1, num_2):
    return num_1 + num_2

result = sum_two_nums(num_2=1)
print(result)
  `}
/>
<Repl code={"SyntaxError: non-default argument follows default argument"} />
<div class="separator" />

<h2>Variable Number of Arguments</h2>
<p>
  Sometimes it is not clear fromt the very beginning how many arguments the user
  is going to provide. We can use <Highlight>*args</Highlight> for a variable number
  of positional arguments and
  <Highlight>**kwargs</Highlight> for a variable number of keyword arguments. The
  names args and kwargs can be replaced by other names (for example *params), but
  this is uncommon. When the function receives arguments during a function call all
  positional arguments are saved in the tuple <Highlight>args</Highlight> and all
  keyword arguments are saved in the dictionary <Highlight>kwargs</Highlight>.
</p>
<Code
  code={`
def variable_args_kwargs(*args, **kwargs):
    print(args) 
    print(kwargs)

variable_args_kwargs(1, 2, 3, creator="World4AI")
  `}
/>
<Repl
  code={`
(1, 2, 3)
{'creator': 'World4AI'}
  `}
/>
<p>
  Oftentimes the variable number of arguments is combined with upacking
  operators <Highlight>*</Highlight> and <Highlight>**</Highlight>.
</p>
<Code
  code={`
def variable_args_kwargs(*args, **kwargs):
    print(args) 
    print(kwargs)

variable_args_kwargs(*[1, 2, 3, 4, 5, 6], **{'creator': 'World4AI', 'year': 2021})
  `}
/>
<Repl
  code={`
(1, 2, 3, 4, 5, 6)
{'creator': 'World4AI', 'year': 2021}
  `}
/>
<div class="separator" />
