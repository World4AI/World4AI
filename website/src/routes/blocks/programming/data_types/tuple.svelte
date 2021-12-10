<script>
  import Question from "$lib/Question.svelte";
  import Operator from "$lib/Operator.svelte";
  import Repl from "$lib/Repl.svelte";
  import Math from "$lib/Math.svelte";
</script>

<h1>Tuple</h1>
<Question>What operators are available for tuples?</Question>
<div class="separator" />

<h2>Operators</h2>
<p>
  Many of the operators that were covered with strings work with tuples in a
  very similar manner. Most of the topics covered here should therefore be very
  familiar.
</p>
<h3>Concatenation (addition)</h3>
<p>
  The addition operators <Operator><Math latex={"+"} /></Operator> and <Operator
    ><Math latex={"+="} /></Operator
  > can be used to concatenate two tuples.
</p>
<Repl
  code={`>>> my_tuple = (1, 2, 3)
>>> my_tuple + (3, 4, 5)
(1, 2, 3, 3, 4, 5)`}
/>
<h3>Repitition (multiplication)</h3>
<p>
  The multiplication operators <Operator><Math latex={"*"} /></Operator> and <Operator
    ><Math latex={"*="} /></Operator
  > repeat the same tuple several times.
</p>
<Repl
  code={`>>> my_tuple = (1, 2, 3)
>>> my_tuple * 2
(1, 2, 3, 1, 2, 3)`}
/>
<h3>Comparisons</h3>
<p>
  Comparison operators like <Operator><Math latex={"=="} /></Operator> or <Operator
    ><Math latex={">"} /></Operator
  > can be applied to tuples. Each individual element is compared piece by piece
  until the first unequal element. If the tuple is larger or lower is determined
  solely by the first unequel element.
</p>
<Repl
  code={`>>> (1, 2, 3) > (1, 2, 2)
True
>>> (1, 2, 'AB') > (1, 2, 'AA')
True
>>> (1, 2, 3, 22) > (1, 1, 3, 1)
True
>>> (1, 2) > (1, 2, 1)
False`}
/>
<h3>Membership</h3>
<p>
  We can check if any an object is contained within the tuple by using the <Operator
    >in</Operator
  > and the <Operator>not in</Operator> operators.
</p>
<Repl
  code={`>>> my_tuple = (1, 2, 3)
>>> 1 in my_tuple
True
>>> 4 not in my_tuple
True`}
/>
<h3>Slicing</h3>
<p>
  Slicing works very similar for strings, tuples and lists. You can convince
  yourself by looking at the example of tuple slicing below.
</p>
<Repl
  code={`>>> my_tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
>>> my_tuple[0:5]
(0, 1, 2, 3, 4)
>>> my_tuple[:5]
(0, 1, 2, 3, 4)
>>> my_tuple[2:6]
(2, 3, 4, 5)
>>> my_tuple[-5:]
(5, 6, 7, 8, 9)
>>> my_tuple[::2]
(0, 2, 4, 6, 8)`}
/>
<h3>Multiple Assignments</h3>
<p>
  We can assign values to several variables at the same time using the
  assignment operator <Operator><Math latex={"="} /></Operator>.
</p>
<Repl
  code={`>>> a, b, c = 0, 1, 2
>>> a
0
>>> b
1
>>> c
2`}
/>
<p>
  Usually the assigment operation with multiple variables is conducted using
  tuples. In that case we call the operation unpacking. Unpacking works in such
  a way where each of the variables gets a "piece" of the tuple.
</p>
<Repl
  code={`>>> a, b, c = (0, 1, 2)
>>> a
0
>>> b
1
>>> c
2`}
/>
<p>
  It is often the case that you receive a tuple, but would like to work with
  individal elements. For that purpose unpacking is a common practice.
</p>
<div class="separator" />
