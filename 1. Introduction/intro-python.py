# %% [markdown]
# # 1: Introduction to Python
# 

# %% [markdown]
# ## Python program files

# %% [markdown]
# * Python code is usually stored in text files with the file ending "`.py`":
# 
#         myprogram.py
# 
# * Every line in a Python program file is assumed to be a Python statement, or part thereof. 
# 
#     * The only exception is comment lines, which start with the character `#` (optionally preceded by an arbitrary number of white-space characters, i.e., tabs or spaces). Comment lines are ignored by the Python interpreter.
# 
# 
# * To run our Python program from the command line, use:
# 
#         $ python myprogram.py
# 
# * On UNIX systems it is common to define the path to the interpreter on the first line of the program (note that this is a comment line as far as the Python interpreter is concerned):
# 
#         #!/usr/bin/env python
# 
#   If we do, and if we additionally set the file script to be executable, we can run the program like this:
# 
#         $ myprogram.py

# %% [markdown]
# ## Jupyter notebooks

# %% [markdown]
# This file - a Jupyter notebook -  does not follow the standard pattern with Python code in a text file. Instead, a Jupyter notebook is stored as a file in the [JSON](http://en.wikipedia.org/wiki/JSON) format. The advantage is that we can mix formatted text, Python code and code output. It requires the Jupyter notebook server to run it though, and therefore isn't a stand-alone Python program as described above. Other than that, there is no difference between the Python code that goes into a program file or a Jupyter notebook.

# %% [markdown]
# Run a code cells using Shift-Enter or pressing the "Run" button in the toolbar above. Jupyter notebook automatically prints out the last made statement in the file. This can be handy for quickly printing output during data analysis.

# %%
x = 1 + 2

x

# %% [markdown]
# ## Contents
# This notebook covers the following aspects of Python:
# 
# 1. Built-in modules
# 2. Variables and types
# 3. Operators and comparisons
# 4. Compound types: Strings, List and dictionaries
# 5. Control Flow
# 6. Loops
# 7. Functions
# 8. Classes
# 9. Modules
# 10. Exceptions

# %% [markdown]
# ## 1. Built-in modules

# %% [markdown]
# Most of the functionality in Python is provided by *modules*. The Python Standard Library is a large collection of modules that provides *cross-platform* implementations of common facilities such as access to the operating system, file I/O, string management, network communication, and much more.

# %% [markdown]
# ### References

# %% [markdown]
#  * [The Python Language Reference]
#  * [The Python Standard Library]
# 
# 
# [The Python Language Reference]: http://docs.python.org/2/reference/index.html
# [The Python Standard Library]: http://docs.python.org/2/library/
# To use a module in a Python program it first has to be imported. A module can be imported using the `import` statement. To import the module `math`, which contains many standard mathematical functions, we can do:

# %%
import math

# %% [markdown]
# This includes the whole module and makes it available for use later in the program. For example, we can write the following program using the math module:

# %%
import math

math.cos(2 * math.pi)

# %% [markdown]
# Alternatively, we can choose to import all symbols (functions and variables) in a module to the current namespace (so that we don't need to use the prefix "`math.`" every time we use something from the `math` module):

# %%
from math import *

cos(2 * pi)

# %% [markdown]
# This pattern can be very convenient, but in large programs that include many modules it is often a good idea to keep the symbols from each module in their own namespaces, by using the `import math` pattern. This would eliminate potentially confusing problems with name space collisions.
# 
# As a third alternative, we can choose to import only a few selected symbols from a module by explicitly listing which ones we want to import instead of using the wildcard character `*`:

# %%
from math import cos, pi

cos(2 * pi)

# %% [markdown]
# ### Looking at what a module contains, and its documentation

# %% [markdown]
# Once a module is imported, we can list the symbols it provides using the `dir` function:

# %%
import math

dir(math)

# %% [markdown]
# And using the function `help` we can get a description of each function (almost .. not all functions have docstrings, as they are technically called, but the vast majority of functions are documented this way). 

# %%
help(math.log)

# %%
log(10)

# %%
log(10, 2)

# %% [markdown]
# We can also use the `help` function directly on modules: Try
# 
#     help(math) 
# 
# Some very useful modules form the Python standard library are `os`, `sys`, `math`, `shutil`, `re`, `subprocess`, `multiprocessing`, `threading`. 
# 
# A complete lists of standard modules for Python 2 and Python 3 are available at http://docs.python.org/2/library/ and http://docs.python.org/3/library/, respectively.

# %%
# use the help function on the math module
help(math)

# %% [markdown]
# ## 2. Variables and types

# %% [markdown]
# ### Symbol names 

# %% [markdown]
# Variable names in Python can contain alphanumerical characters `a-z`, `A-Z`, `0-9` and some special characters such as `_`. Normal variable names must start with a letter. 
# 
# By convention, variable names start with a lower-case letter, and Class names start with a capital letter. Another convention is to use the lower dash `_` to separate words, instead of camelcase. E.g. instead of `myVariable`, we tend to write `my_variable`. For further code style guidlines, you can check out Google's style guides for python: http://google.github.io/styleguide/pyguide.html
# 
# In addition, there are a number of Python keywords that cannot be used as variable names. These keywords are:
# 
#     and, as, assert, break, class, continue, def, del, elif, else, except, 
#     exec, finally, for, from, global, if, import, in, is, lambda, not, or,
#     pass, print, raise, return, try, while, with, yield

# %% [markdown]
# ### Assignment

# %% [markdown]
# 
# 
# The assignment operator in Python is `=`. Python is a dynamically typed language, so we do not need to specify the type of a variable when we create one.
# 
# Assigning a value to a new variable creates the variable:

# %%
# variable assignments
x = 1.0
my_variable = 12.2

# %% [markdown]
# Although not explicitly specified, a variable does have a type associated with it. The type is derived from the value it was assigned.

# %%
x = 1
type(x)

# %% [markdown]
# If we assign a new value to a variable, its type can change.

# %%
x =[1,2,3,4,5]
type(x)

# %% [markdown]
# If we try to use a variable that has not yet been defined we get a `NameError`:

# %%
y

# %% [markdown]
# ### Fundamental types

# %% [markdown]
# Examples of fundamental types in Python are shown below.

# %%
# integers
type(1)

# %%
# float
type(1.0)

# %%
# boolean
type(True)

# %%
# complex numbers: note the use of `j` to specify the imaginary part
type(1.0 - 1.0j)

# %% [markdown]
# ### Type casting

# %% [markdown]
# Examples of type casting in Python are shown below.

# %%
x = 1.5

print(x, type(x))

# %%
x = int(x)

print(x, type(x))

# %%
z = complex(x)

print(z, type(z))

# %%
x = float(z)

# %% [markdown]
# Complex variables cannot be cast to floats or integers. We need to use `z.real` or `z.imag` to extract the part of the complex number we want:

# %%
y = bool(z.real)

print(z.real, " -> ", y, type(y))

y = bool(z.imag)

print(z.imag, " -> ", y, type(y))

# %% [markdown]
# ## 3. Operators and comparisons

# %% [markdown]
# Most operators and comparisons in Python work as one would expect:
# 
# * Arithmetic operators `+`, `-`, `*`, `/`, `//` (integer division), `%` (modulo), `**` power
# 

# %%
1 + 2, 1 - 2, 1 * 2, 1 / 2

# %%
1.0 + 2.0, 1.0 - 2.0, 1.0 * 2.0, 1.0 / 2.0

# %%
# Integer division of float and integer numbers
5.0 // 2.0, 5 // 2

# %%
# Modulo of float and integer numbers
5.0 % 2.0, 5 % 2

# %%
# Note! The power operator in python isn't ^, but ** !
2 ** 10

# %% [markdown]
# * The boolean operators are spelled out as words: `and`, `not`, `or`. 

# %%
True and False

# %%
not False

# %%
True or False

# %% [markdown]
# * Comparison operators `>`, `<`, `>=` (greater or equal), `<=` (less or equal), `==` (equality), `is` (identical).

# %%
2 > 1, 2 < 1

# %%
2 > 2, 2 < 2

# %%
2 >= 2, 2 <= 2

# %%
# equality
[1,2] == [1,2]

# %%
# objects identical (sae memory adress)?
o1 = [1,2]
o2 = [1,2]

o1 is o2

# %% [markdown]
# ## 4. Compound types: Strings, List and dictionaries

# %% [markdown]
# ### Strings

# %% [markdown]
# Strings are the variable type that is used for storing text.

# %%
s = "Hello world"
type(s)

# %%
str.capitalize('abcd')

# %%
# length of the string: the number of characters including spaces
len(s)

# %%
# replace a substring in a string with something else
s2 = s.replace("world", "test")
print(s2)

# %% [markdown]
# We can index a character in a string using `[]`:

# %%
s[0]

# %% [markdown]
# **Heads up MATLAB users:** Indexing start at 0!
# 
# We can extract a part of a string using the syntax `[start:stop]`, which extracts characters between index `start` and `stop`, including the `start` element and excluding the `stop` element: [start, stop).

# %%
s[0:5]

# %% [markdown]
# If we do not define `start` or `stop` in `[start:stop]`, e.g. `[:stop]`, `start` will default to 0 and `end` to the end of the string.

# %%
s[:5]

# %%
s[6:]

# %%
s[:]

# %% [markdown]
# We can also define the step size using the syntax `[start:end:step]` (the default value for `step` is 1, as we saw above). This technique is called *slicing*.

# %%
s[::1]

# %%
s[::2]

# %% [markdown]
# #### String formatting examples

# %%
print("str1", "str2", "str3")  # The print statement concatenates strings with a space

# %%
print("str1", 1.0, False, -1j)  # The print statements converts all arguments to strings

# %%
print("str1" + "str2" + "str3") # strings added with + are concatenated without space

# %%
print("value = %f" % 1.0)       # we can use C-style string formatting

# %%
# this formatting creates a string
s2 = "value1 = %.2f. value2 = %d" % (3.1415, 1.5)

print(s2)

# %%
# alternative, more intuitive way of formatting a string 
s3 = 'value1 = {0}, value2 = {1}'.format(3.1415, 1.5)

print(s3)

# %% [markdown]
# ### Lists

# %% [markdown]
# Lists are very similar to strings, except that each element can be of any type.
# 
# The syntax for creating lists in Python is `[value_1,value_2, ... ,value_n]`:

# %%
l = [1,2,3,4]

print(type(l))
print(l)

# %% [markdown]
# We can use the same slicing techniques to manipulate lists as we could use on strings:

# %%
print(l)
print(l[1:3])
print(l[::2])

# %% [markdown]
# Elements in a list do not all have to be of the same type. However, to avoid errors, it is advised to store only values of one type in a list:

# %%
l = [1, 'a', 1.0, 1-1j]

print(l)

# %% [markdown]
# Python lists can be inhomogeneous and arbitrarily nested:

# %%
nested_list = [1, [2, [3, [4, [5]]]]]
nested_list

# %% [markdown]
# Lists play a very important role in Python, and are used in loops and other flow control structures (discussed below). There are a number of convenient functions for generating lists of various types, for example the `range` function:

# %%
start = 10
stop = 30
step = 2

range(start, stop, step)

# %%
# in python 3 range generates an iterator, which can be converted to a list using 'list(...)'.
list(range(start, stop, step))

# %%
list(range(-10, 10))

# %%
s = 'Hello world'
s

# %%
# convert a string to a list by type casting:
s2 = list(s)

s2

# %%
# sorting lists, uppercase and lowercase characters are sorted separately
s2.sort()

print(s2)

# %% [markdown]
# #### Adding, inserting, modifying, and removing elements from lists

# %%
# create a new empty list
l = []

# add an elements using `append`
l.append("A")
l.append("d")
l.append("d")

print(l)

# %% [markdown]
# We can modify lists by assigning new values to elements in the list. In technical jargon, lists are *mutable*.

# %%
l[1] = "p"
l[2] = "p"

print(l)

# %% [markdown]
# Insert an element at an specific index using `insert`

# %%
l.insert(0, "i")
l.insert(1, "n")
l.insert(2, "s")
l.insert(3, "e")
l.insert(4, "r")
l.insert(5, "t")

print(l)

# %% [markdown]
# Remove first element with specific value using 'remove'

# %%
l.remove("A")

print(l)

# %% [markdown]
# Remove an element at a specific location using `del`:

# %%
del l[7]
del l[6]

print(l)

# %% [markdown]
# See `help(list)` for more details, or read the online documentation 

# %% [markdown]
# ### Tuples

# %% [markdown]
# Tuples are like lists, except that they cannot be modified once created, that is they are *immutable*. 
# 
# In Python, tuples are created using the syntax `(..., ..., ...)`:

# %%
point = (10, 20)

print(point, type(point))

# %% [markdown]
# We can unpack a tuple by assigning it to comma-separated variables:

# %%
x, y = point

print("x =", x)
print("y =", y)

# %% [markdown]
# ### Dictionaries

# %% [markdown]
# Dictionaries are also like Java(script) Objects. The values are connected to keys of the dictionary. The keys can be used to retrieve values from a dictionary. The syntax for dictionaries is `{key1 : value1, ...}`:

# %%
 values = {
     "key1" : 1.0,
     "key2" : 2,
     "key3" : [1,2,3]
 }

print(type(values))
print(values)

# %%
# Retrieve a value by using the key
values["key1"]

# %%
print("key1 = ", values["key1"])
print("key2 = ", values["key2"])
print("key3 = ", values["key3"])

# %%
# You can reassign the values corresponding to a key
values["key1"] = "A"
values["key2"] = "B"

# Assigning a value to an unknown key results in the creation of that key
values["key4"] = "D"

print("key1 = ", values["key1"])
print("key2 = ", values["key2"])
print("key3 = ", values["key3"])
print("key4 = ", values["key4"])

# %% [markdown]
# ## 5. Control Flow

# %% [markdown]
# ### Conditional statements: if, elif, else

# %% [markdown]
# The Python syntax for conditional execution of code use the keywords `if`, `elif` (else if), `else`:

# %%
statement1 = False
statement2 = False

if statement1:
    print("statement1 is True")
    
elif statement2:
    print("statement2 is True")
    
else:
    print("statement1 and statement2 are False")

# %% [markdown]
# As noted at the top of this notebook, we encounter an unusual aspect of the Python programming language: Program blocks are defined by their indentation level. 
# 
# Compare to the equivalent Java code:
# 
#     if (statement1) {
#         System.out.println("statement1 is True");
#     } else if (statement2) {
#         System.out.println("statement2 is True");
#     } else {
#         System.out.println("statement1 and statement2 are False");
#     }
# 
# In Java blocks are defined by the enclosing curly brakets `{` and `}`. And the level of indentation (white space before the code statements) does not matter (completely optional). 
# 
# But in Python, the extent of a code block is defined by the indentation level (usually a tab or two/four white spaces, which is automatically detected by the interpreter). This means that we have to be careful to indent our code correctly, or else we will get syntax errors. 

# %% [markdown]
# #### Examples:

# %%
statement1 = True
statement2 = True

if statement1:
    if not statement2:
        # should not print, since statement2 == True
        print("both statement1 and statement2 are True")

# %%
# Bad indentation!
if statement1:
    if not statement2:
    print("both statement1 and statement2 are True")  # this line is not properly indented

# %%
statement1 = True 

if statement1:
    print("printed if statement1 is True")
    
    print("still inside the if block")

# %%
statement1 = False

if statement1:
    print("printed if statement1 is True")
    
print("now outside the if block")

# %% [markdown]
# ## 6. Loops

# %% [markdown]
# In Python, loops can be programmed in a number of different ways. The most common is the `for` loop, which is used together with iterable objects with `in`, such as lists. The basic syntax is:

# %% [markdown]
# ### **`for` loops**:

# %%
for x in [1,2,3]:
    print(x)

# %% [markdown]
# The `for` loop iterates over the elements of the supplied list, and executes the containing block once for each element. Any kind of list can be used in the `for` loop. For example:

# %%
for x in range(4): # by default range start at 0
    print(x)

# %% [markdown]
# Note that `list(range(4))` returns a _list_ with the elements `[0, 1, 2, 3]`, it does not include 4!

# %%
for x in range(-3,3):
    print(x)

# %%
for word in ["scientific", "computing", "with", "python"]:
    print(word)

# %% [markdown]
# To iterate over key-value pairs of a dictionary one can also jus use `in`:

# %%
 values = {
     "key1" : 1.0,
     "key2" : 2,
     "key3" : [1,2,3]
 }

for key, value in values.items():
    print(key + " = " + str(value))

# %% [markdown]
# Sometimes it is useful to have access to the indices of the values when iterating. We can use the `enumerate` function for this:

# %%
for index, x in enumerate(range(-3,3)):
    print(index, x)

# %% [markdown]
# ### List comprehensions: Creating lists using `for` loops:

# %% [markdown]
# A convenient and compact way to initialize lists can be done with `for` loops:

# %%
l1 = [x**2 for x in range(0,5)]

print(l1)

# %% [markdown]
# ### `while` loops:

# %%
i = 0

while i < 5:
    print(i)
    i = i + 1
    
print("done", i)

# %% [markdown]
# Note that the `print("done", i)` statement is not part of the `while` loop body because of the difference in indentation.

# %% [markdown]
# ## 7. Functions

# %% [markdown]
# A function (method in Java) in Python is defined using the keyword `def`, followed by a function name, a signature (lists the parameters) within parentheses `()`, and a colon `:`. The following code, with one additional level of indentation, is the function body.

# %%
def func0():   
    print("func0 is called")

# %%
func0()

# %% [markdown]
# Optionally, but highly recommended, we can define a so called "docstring", which is a description of the function's purpose and behaivor. The docstring should follow directly after the function definition, before the code in the function body.

# %%
def func1(s):
    """
    Print a string 's' and tell how many characters it has    
    """
    
    print(s + " has " + str(len(s)) + " characters")

# %%
help(func1)

# %%
func1("test")

# %% [markdown]
# Functions that return a value use the `return` keyword:

# %%
def square(x):
    """
    Return the square of x.
    """
    return x ** 2

# %%
square(4)

# %% [markdown]
# We can return multiple values from a function using tuples (see above):

# %%
def powers(x):
    """
    Return a few powers of x.
    """
    return x ** 2, x ** 3, x ** 4

# %%
powers(3)

# %%
x2, x3, x4 = powers(3)

print(x3)

# %% [markdown]
# ### Default argument and keyword arguments

# %% [markdown]
# In a definition of a function, we can give default values to the arguments the function takes:

# %%
def myfunc(x, p=2, debug=False):
    if debug:
        print("evaluating myfunc for x = " + str(x) + " using exponent p = " + str(p))
    return x**p

# %% [markdown]
# If we don't provide a value of the `debug` argument when calling the the function `myfunc` it defaults to the value provided in the function definition:

# %%
myfunc(5)

# %%
myfunc(5, debug=True)

# %% [markdown]
# If we explicitly list the name of the arguments in the function calls, they do not need to come in the same order as in the function definition. This is called *keyword* arguments, and is often very useful in functions that takes a lot of optional arguments.

# %%
myfunc(p=3, debug=True, x=7)

# %% [markdown]
# ### Unnamed functions (lambda function)

# %% [markdown]
# In Python we can also create unnamed functions, using the `lambda` keyword:

# %%
f1 = lambda x: x**2
    
# is equivalent to 

def f2(x):
    return x**2

# %%
f1(2), f2(2)

# %% [markdown]
# This technique is useful for example when we want to pass a simple function as an argument to another function, like this:

# %%
# map is a built-in python function
map(lambda x: x**2, range(-3,4))

# %%
# in python 3 we can use `list(...)` to convert the iterator to an explicit list
list(map(lambda x: x**2, range(-3,4)))

# %% [markdown]
# ## 8. Classes

# %% [markdown]
# Classes are the key features of object-oriented programming. A class is a structure for representing an object and the operations that can be performed on the object. 
# 
# In Python a class can contain *attributes* (variables) and *methods* (functions).
# 
# A class is defined almost like a function, but using the `class` keyword, and the class definition usually contains a number of class method definitions (a function in a class).
# 
# * Each class method should have an argument `self` as it first argument. This object is a self-reference.
# 
# * Some class method names have special meaning, for example:
# 
#     * `__init__`: The name of the method that is invoked when the object is first created.
#     * `__str__` : A method that is invoked when a simple string representation of the class is needed, as for example when printed.
#     * There are many more

# %%
class Point:
    """
    Simple class for representing a point in a Cartesian coordinate system.
    """
    
    def __init__(self, x, y):
        """
        Create a new Point at x, y.
        """
        self.x = x
        self.y = y
        
    def translate(self, dx, dy):
        """
        Translate the point by dx and dy in the x and y direction.
        """
        self.x += dx
        self.y += dy
        
    def __str__(self):
        return("Point at [%f, %f]" % (self.x, self.y))

# %% [markdown]
# To create a new instance of a class:

# %%
p1 = Point(0, 0) # this will invoke the __init__ method in the Point class

print(p1)         # this will invoke the __str__ method

# %% [markdown]
# To invoke a class method in the class instance `p`:

# %%
print(p1)
p1.translate(0.25, 1.5)
print(p1)

# %% [markdown]
# Note that calling class methods can modify the state of that particular class instance, but does not effect other class instances or any global variables.
# 
# That is one of the nice things about object-oriented design: code such as functions and related variables are grouped in separate and independent entities. 

# %% [markdown]
# ## 9. Modules

# %% [markdown]
# One of the most important concepts in good programming is to reuse code and avoid repetitions.
# 
# The idea is to write functions and classes with a well-defined purpose and scope, and reuse these instead of repeating similar code in different part of a program (modular programming). The result is usually that readability and maintainability of a program is greatly improved. What this means in practice is that our programs have fewer bugs, are easier to extend and debug/troubleshoot. 
# 
# Python supports modular programming at different levels. Functions and classes are examples of tools for low-level modular programming. Python modules are a higher-level modular programming construct, where we can collect related variables, functions and classes in a module. A Python module is defined in a Python file (with file-ending `.py`), and it can be made accessible to other Python modules and programs using the `import` statement. 
# 
# Consider the following example: the file `mymodule.py` contains simple example implementations of a variable, function and a class:

# %%
%%file mymodule.py
"""
Example of a Python module. Contains a variable called my_variable,
a function called my_function, and a class called MyClass.
"""

my_variable = 0

def my_function():
    """
    Example function
    """
    return my_variable
    
class MyClass:
    """
    Example class.
    """

    def __init__(self):
        self.variable = my_variable
        
    def set_variable(self, new_value):
        """
        Set self.variable to a new value
        """
        self.variable = new_value
        
    def get_variable(self):
        return self.variable

# %% [markdown]
# We can import the module `mymodule` into our Python program using `import`:

# %%
%load_ext autoreload
%autoreload 2 # This makes sure all modules are reloaded every time before executing the Python code typed.

import mymodule

# %% [markdown]
# Use `help(module)` to get a summary of what the module provides:

# %%
help(mymodule)

# %%
mymodule.my_variable

# %%
mymodule.my_function() 

# %%
my_class = mymodule.MyClass() 
my_class.set_variable(10)
my_class.get_variable()

# %% [markdown]
# ## 10. Exceptions

# %% [markdown]
# In Python errors are managed with a special language construct called "Exceptions". When errors occur exceptions can be raised, which interrupts the normal program flow and fallback to somewhere else in the code where the closest try-except statement is defined.

# %% [markdown]
# To generate an exception we can use the `raise` statement, which takes an argument that must be an instance of the class `BaseException` or a class derived from it. 

# %%
raise Exception("description of the error")

# %% [markdown]
# A typical use of exceptions is to abort functions when some error condition occurs, for example:
# 
#     def my_function(arguments):
#     
#         if not verify(arguments):
#             raise Exception("Invalid arguments")
#         
#         # rest of the code goes here

# %% [markdown]
# Congratulations! You now know the most important ins and outs of Python 3! Next step is to do the tutorial on **NumPy**, the library which you will very often use during this course!


