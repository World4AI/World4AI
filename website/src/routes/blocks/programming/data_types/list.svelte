<script>
  import Question from "$lib/Question.svelte";
  import Operator from "$lib/Operator.svelte";
  import Math from "$lib/Math.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | List Datatype</title>
  <meta
    name="description"
    content="A list is a mutable collection of objects. Python provides several operators and methods to access and process lists."
  />
</svelte:head>

<h1>List</h1>
<Question>How can we work with lists in Python?</Question>
<div class="separator" />

<p>
  A list is a mutable collection of objects. This built-in data structure is
  often used in machine learning. For example to keep track of the performance
  of a learning algoirthm. Whenever you assume, that the collection of data is
  going to grow, to shrink or to change, you should prefer a list over a tuple.
</p>
<p>
  The operators that work with strings and tuples work with lists in a very
  similar way, therefore the topics that are going to be covered in the
  operators sections will seem a little redundant. The main difference is the
  option to change the value of a list at a certain index.
</p>
<p>
  A list is a mutable object, therefore Python additionally provides several
  methods that we can utilize to update, remove and insert data.
</p>
<div class="separator" />
<h2>Operators</h2>
<h3>Concatenation (Addition)</h3>
<p>
  Just as with strings and tuples the addition operators <Operator
    ><Math latex={"+"} /></Operator
  > and <Operator><Math latex={"+="} /></Operator> concatenate lists.
</p>
<Repl
  code={`>>> [1, 2, 3] + [4, 5]
[1, 2, 3, 4, 5]`}
/>
<h3>Repetition (Multiplication)</h3>
<p>
  The multiplication operators <Operator><Math latex={"*"} /></Operator> and <Operator
    ><Math latex={"*="} /></Operator
  > repeat the same list several times.
</p>
<Repl
  code={`>>> [1, 2, 3] * 3
[1, 2, 3, 1, 2, 3, 1, 2, 3]`}
/>
<h3>Comparisons</h3>
<p>
  The comparison operators like <Operator><Math latex={"="} /></Operator> or <Operator
    ><Math latex={">"} /></Operator
  >compare individual elements of the list. If all the elements are equal then
  both lists are considered equal. Otherwise the difference is determined based
  on the first unequal element in the lists.
</p>
<Repl
  code={`>>> [1, 2, 3] > [1, 2, 2]
True
>>> [1] > [2]
False
>>> ['a', 'b', 'c'] < ['b']
True`}
/>
<h3>Membership</h3>
<p>
  With the <Operator>in</Operator> and <Operator>not in</Operator> operators we can
  determine if an object with a certain value is contained within a list.
</p>
<Repl
  code={`>>> 1 in [5, 6, 7]
False
>>> 5 in [5, 6, 7]
True
>>> 5 not in [5, 6, 7]
False`}
/>
<h3>Slicing</h3>
<p>
  The way slicing works is almost identical for lists, tuples and strings. Below
  is an example of list slicing.
</p>
<Repl
  code={`>>> my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> my_list[0:5]
[0, 1, 2, 3, 4]
>>> my_list[5:]
[5, 6, 7, 8, 9]
>>> my_list[-2:]
[8, 9]
>>> my_list[0:6:2]
[0, 2, 4]`}
/>
<p>Additioanly we can use slicing to change the values of the list.</p>
<Repl
  code={`>>> my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> id(my_list)
140644866899520
>>> my_list[:2] = ['a', 'b']
>>> my_list
['a', 'b', 2, 3, 4, 5, 6, 7, 8, 9]
>>> id(my_list)
140644866899520`}
/>
<div class="separator" />

<h2>Methods</h2>
<h3>append</h3>
<p>The append() method appends an object to the end of the list.</p>
<Repl
  code={`>>> my_list = [1, 2, 3]
>>> my_list.append(4)
>>> my_list
[1, 2, 3, 4]`}
/>
<h3>clear</h3>
<p>The clear() method empties the list.</p>
<Repl
  code={`>>> my_list = [1, 2, 3]
>>> my_list.clear()
>>> my_list
[]
>>>`}
/>
<h3>copy</h3>
<p>
  When we assign an existing list to a variable that does not create a copy of
  the list.
</p>
<Repl
  code={`>>> a = [1, 2, 3]
>>> b = a
>>> a[0] = 99
>>> a
[99, 2, 3]
>>> b
[99, 2, 3]
>>>`}
/>
<p>The copy() method actually copies an object.</p>
<Repl
  code={`>>> my_list_1 = [1, 2, 3]
>>> my_list_2 = my_list_1.copy()
>>> my_list_1
[1, 2, 3]
>>> my_list_2
[1, 2, 3]
>>> my_list_1[0] = 10
>>> my_list_1
[10, 2, 3]
>>> my_list_2
[1, 2, 3]`}
/>
<p>
  The copy() method creates a shallow copy. In a shallow copy a list within a
  list is not copied.
</p>
<Repl
  code={`>>> my_list_1 = [1, [2,3], 4, 5]
>>> my_list_2 = my_list_1.copy()
>>> my_list_2[1][0] = 99
>>> my_list_1
[1, [99, 3], 4, 5]
>>> my_list_2
[1, [99, 3], 4, 5]`}
/>
<h3>count</h3>
<p>
  The count() method counts the occurence of a certain object within a list.
</p>
<Repl
  code={`>>> my_list = [1, 1, 2, 3]
>>> my_list.count(1)
2
>>> my_list.count(2)
1`}
/>
<h3>extend</h3>
<p>The extend method concatenates an different oject to an object.</p>
<Repl
  code={`>>> my_list_1 = [1, 2, 3]
>>> my_list_2 = ['a', 'b', 'c']
>>> my_list_1.extend(my_list_2)
>>> my_list_1
[1, 2, 3, 'a', 'b', 'c']`}
/>
<h3>index</h3>
<p>The index() method returns the index of an object within a list.</p>
<Repl
  code={`>>> my_list = [1, 2, 3, 4]
>>> my_list.index(2)
1
>>> my_list.index(4)
3`}
/>
<h3>insert</h3>
<p>
  The insert() method works similar to the append() method. It allows to add an
  object to an already existing list, but you can specify the position where the
  oject should be inserted. The method is defined as follows: insert(position,
  object).
</p>
<Repl
  code={`>>> my_list = [0, 1, 2, 3]
>>> my_list.insert(0, 22)
>>> my_list
[22, 0, 1, 2, 3]
>>> my_list.insert(3, 77)
>>> my_list
[22, 0, 1, 77, 2, 3]`}
/>
<h3>pop</h3>
<p>
  The pop() method removes the last element from the list and returns the
  removed element. You can also specify the exact index of the element that you
  want to remove. For example [0,1,2].pop(0) would remove and return the value
  0.
</p>
<Repl
  code={`>>> my_list = [0, 1, 2, 3, 4, 5]
>>> popped_value = my_list.pop()
>>> popped_value
5
>>> my_list
[0, 1, 2, 3, 4]
>>> my_list.pop(0)
0
>>> my_list
[1, 2, 3, 4]`}
/>
<h3>remove</h3>
<p>The remove() method removes a specified value from a list.</p>
<Repl
  code={`>>> my_list = [1, 2, 3, 4]
>>> my_list.remove(3)
>>> my_list
[1, 2, 4]`}
/>
<h3>reverse</h3>
<p>The reverse() method reverses the original list in place.</p>
<Repl
  code={`>>> my_list = [1, 2, 3, 4]
>>> my_list.reverse()
>>> my_list
[4, 3, 2, 1]`}
/>
<h3>sort</h3>
<p>
  The sort() method changes the list by sorting the list in ascending order.
</p>
<Repl
  code={`>>> my_list = [3, 2, 1, 5, 3, 11]
>>> my_list.sort()
>>> my_list
[1, 2, 3, 3, 5, 11]`}
/>
<p>We can utilize the "reverse" option to sort the list in reversed order</p>
<Repl
  code={`>>> my_list = [3, 2, 1, 5, 3, 11]
>>> my_list.sort(reverse=True)
>>> my_list
[11, 5, 3, 3, 2, 1]`}
/>
<div class="separator" />
