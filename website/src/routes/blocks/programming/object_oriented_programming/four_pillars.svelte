<script>
  import Question from "$lib/Question.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Code from "$lib/Code.svelte";
  import Repl from "$lib/Repl.svelte";
</script>

<svelte:head>
  <title>World4AI | Programming | Four Pillars of OOP</title>
  <meta
    name="description"
    content="The object oriented programming is build upon 4 pillars: Abstraction, Inheritance, Encapsulation and Polymorphism."
  />
</svelte:head>

<h1>The Four Pillars of Object-Oriented Programming</h1>
<Question
  >What are the four pillars of oop and what does each of the pillars mean?</Question
>
<div class="separator" />
<p>
  When you read about object-oriented programming sooner or later you will
  discover that there are certain properties that an object-oriented programming
  language displays. Those properties are: abstraction, inheritance,
  encapsulation and polymorphism and are often called <Highlight
    >the four pillars of oop</Highlight
  >.
</p>
<div class="separator" />

<h2>Abstraction</h2>
<p>
  The key concept of abstraction is to separate the often complex implementation
  of a class from the interface that the user interacts with. The user does not
  have to deal with the complexity of the calculations and only needs to
  remember the api of the class.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print_coordinates(self):
        print(f'The x coordinate is: {self.x}')
        print(f'The y coordinate is: {self.y}')

coordinates = Coordinates(10, 5)
coordinates.print_coordinates()
  `}
/>
<p>
  The Coordinates class provides the print_coordinates() method as an interface
  to the user. How exactly the method is implemented is irrelevant. The relevant
  part is that the function exists and can be called to print the x and y
  coordinates. This is a relatively simple example, but often we have to work
  with complex external libraries that provide classes with a particular set of
  attributes and methods that we can access. We do not need to spend a lot of
  time looking at the implementation details, but can use the abstract idea of
  what those methods are supposed to represent.
</p>
<p>
  When we start working with Python libraries that are designed to train neural
  networks you will encounter many methods that represent an abstract idea that
  you will know from the theory of deep learning. The libraries are extremely
  complex on the inside and were designed by hundreds of developers who tried to
  optimize the performance and ease of use of those libraries. The interface of
  those libraries is on the other hand extremely intuitive and lets you do
  machine learning research without knowing the the ins and outs of the
  libraries.
</p>
<div class="separator" />

<h2>Inheritance</h2>
<p>
  Often we want to have classes that share some basic functionality. For that
  purpose we can create a parent class, from which the child classes inherit all
  methods and attributes.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print_coordinates(self):
        print(f'The x coordinate is: {self.x}')
        print(f'The y coordinate is: {self.y}')

class Circle(Coordinates):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self.radius = radius

    def print_radius(self):
        print(f'The radius is: {self.radius}')

circle = Circle(x=10, y=5, radius=5)
circle.print_coordinates()
circle.print_radius()
  `}
/>
<Repl
  code={`
The x coordinate is: 10
The y coordinate is: 5
The radius is: 5
  `}
/>
<p>
  Above we implement the Circle class. A cirlce has a set of coordinates, but
  additionally we also need a radius in order to be able to draw the circle. The
  x and y coordinates are already implemented in the Coordinates class, the
  radius will require additional code.
</p>
<p>
  When we write <Highlight>class Circle(Coordinates):</Highlight> that signifies
  that the Circle class inherits all methods and attributes from Coordinates. The
  __init__ method needs three parameters: x, y and the radius. We have already implemented
  the __init__ method in the Coordinates class, which takes care of x and y. When
  we utilize super(), we tell Python, that we want to use some functionality from
  the base class, in this case we use the __init__ method in the of Coordinates to
  set the initial values. That leaves the Circle class to deal with the radius, while
  the base Coordinates class takes care of x and y.
</p>
<div class="separator" />

<h2>Encapsulation</h2>
<p>
  The developers of classes often need to protect the users from themselves. The
  user might try to use the object in such a way that would contradict the
  expected behaviour. To avoid the catastrophic failure of the program,
  developers implement the logic of methods in such a way that data (attributes)
  can be only manipulated in the desired way.
</p>
<p>
  The technique that developers of libraries use for that purpose is called
  encapsulation. Some of the data in a class is declared as private. Only the
  object itself can manipulate that private data. If the user of the library
  needs to change the state of the object he will need to use specific methods
  that the developers of libraries provide as the interface. Python does not
  provide a direct way to create private variables that can not be accessed from
  outside the object. A common convention to declare a variable as private is to
  use a name that starts with an uderscore. For example instead of using the
  attribute x in the coordinate class we will use _x, to let other developers
  know that they should not touch the value directly.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self._x = x
        self._y = y
  `}
/>
<p>
  The problem that we face with the above class is the lack of a way to interact
  with the coordinates. An obvious way to allow access to those variables would
  be to provide methods which allow access to certain attributes. In oop
  programming language we call those methods getters and setters. A getter
  allows us to get some attribute from the object, while a setter allows us to
  set some attribute in an object.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_x(self):
        return self._x

    def set_x(self, x): 
        if x >= 0:
            self._x = x

    def get_y(self):
        return self._y

    def set_x(self, y): 
        if x >= 0:
            self._y = y
  `}
/>
<p>
  Let us observe the setter set_x to get an intuitive understanding why
  encapsulation makes sense. If we assume that we are creating a computer game
  and allow only positive coordinates, some calculations might break down if a
  user tries to use negative numbers. For that reason we always check first if
  the new value for x is positive or 0, if it is not we do not change the value.
</p>
<p>
  Python provides some additional conveniences to work with getters and setters.
  If would for example would be much easier to use coordinates.x instead of
  coordinates.get_x() to get the current value and to use coordinates.x = 10 to
  set the carrent value.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x): 
        if x >= 0:
            self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y): 
        if y >= 0:
            self._y = y

    def print_coordinates(self):
        print(f'The x coordinate is: {self._x}')
        print(f'The y coordinate is: {self._y}')


coordinates = Coordinates(10, 5)
coordinates.print_coordinates()

coordinates.x = 15
coordinates.print_coordinates()
  `}
/>
<p>
  We create two functions for each attribute and name those functions the way we
  would like to access them. For example the function x() will allow us to use
  coordinates.x later. We provide two versions: one for the getter that only
  takes the instance as the input and one for the setter that additionally takes
  the new value. To make those functions accessible the way attributes are, we
  use decorators. Decorators are going to be covered in detail in a later
  chapter, but essentially the goal of a decorator is to adjust the way a method
  or function works. The @property decorator converts a method into a getter.
  The @name_of_attribute.setter decorator converts a method into a setter. Those
  decorators let us access those methods as if they were attributes.
</p>
<div class="separator" />

<h2>Polymorphism</h2>
<p>
  Polymorphism allows to us to be indifferent to whether we are dealing with the
  base class or subclass when calling a base method. This definition is a little
  abstract, therefore we will demonstrate the idea with some examples.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def print_coordinates(self):
        print(f'The x coordinate is: {self._x}')
        print(f'The y coordinate is: {self._y}')

class Circle(Coordinates):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self._radius = radius

    def print_coordinates(self):
        super().print_coordinates()
        print(f'The radius is: {self._radius}')

class Square(Coordinates):
    def __init__(self, x, y, size):
        super().__init__(x, y)
        self._size = size

    def print_coordinates(self):
        super().print_coordinates()
        print(f'The size is: {self._size}')

coordinates = Coordinates(10, 5)
circle = Circle(3, 2, 5)
square = Square(3, 4, 9)

collection = [coordinates, circle, square]

for obj in collection:
    print('-'*30)
    obj.print_coordinates()
  `}
/>
<Repl
  code={`
------------------------------
The x coordinate is: 10
The y coordinate is: 5
------------------------------
The x coordinate is: 3
The y coordinate is: 2
The radius is: 5
------------------------------
The x coordinate is: 3
The y coordinate is: 4
The size is: 9
  `}
/>
<p>
  Above we have the Coordinates class, the Circle class and the Square class. We
  create an instance for each of the classes and put them into a list that we
  loop over. The obj variable can take different shapes (polymorphism means many
  shaped) and call a function with the same name that belongs to classes.
</p>
<p>
  Unlike many other programming languages like C++ or C#, Python uses so called
  duck typing. In C++ the object in a list need to belong to the same base class
  in order to iterate over them in a list and to call the function with the same
  name. Duck typing states: "If it swims like a duck and quacks like a duck, it
  must be a duck". As long as a class implements a certain method, for example
  print_coordinates(), Python does not care what class the object belongs to. If
  Python is able to call the method, the code will be executed.
</p>
<Code
  code={`
class Coordinates:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def print_coordinates(self):
        print(f'The x coordinate is: {self._x}')
        print(f'The y coordinate is: {self._y}')

class Circle(Coordinates):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self._radius = radius

    def print_coordinates(self):
        super().print_coordinates()
        print(f'The radius is: {self._radius}')

class Square(Coordinates):
    def __init__(self, x, y, size):
        super().__init__(x, y)
        self._size = size

    def print_coordinates(self):
        super().print_coordinates()
        print(f'The size is: {self._size}')

class Duck:

    def print_coordinates(self):
        print("QUACK")


coordinates = Coordinates(10, 5)
circle = Circle(3, 2, 5)
square = Square(3, 4, 9)
duck = Duck()

collection = [coordinates, circle, square, duck]

for obj in collection:
    print('-'*30)
    obj.print_coordinates()
  `}
/>
<Repl
  code={`
------------------------------
The x coordinate is: 10
The y coordinate is: 5
------------------------------
The x coordinate is: 3
The y coordinate is: 2
The radius is: 5
------------------------------
The x coordinate is: 3
The y coordinate is: 4
The size is: 9
------------------------------
QUACK
  `}
/>
<p>
  Above we can see the application of duck typing. The class Duck has not
  relation to any other class, yet we have no problem calling the
  print_coordinates() function.
</p>
<div class="separator" />
