# %% [markdown]
# # 1: Introduction to NumPy

# %% [markdown]
# ## Step 1: Array programming with NumPy

# %%
import numpy as np

# %% [markdown]
# ### Declaring a regular Python list

# %%
list1 = [1, 2, 3, 4]
type(list1)

# %% [markdown]
# ### Making a numpy array using Python lists

# %%
array1 = np.array(list1)
array1

# %%
# The type of array1 is an ndarray: an n-dimensional array
type(array1)

# %%
print(array1)

# %% [markdown]
# ### We can also use more dimensions

# %%
# Declare an extra list.
list2 = [11, 22, 33, 44]
# Combine the lists into a 2-dimensional list.
lists = [list1, list2]
lists

# %%
# make a 2-dimensional NumPy array
array2 = np.array(lists)
array2

# %% [markdown]
# ### And print their shapes
# We would obviously expect a (4,1) and a (4,2)... or don't we?

# %%
print("Arr1: ", array1.shape)
print("Arr2: ", array2.shape)

# %% [markdown]
# The best way to think about NumPy arrays is that they consist of two parts, a _data buffer_ which is just a block of raw elements, and a _view_ which describes how to interpret the data buffer.
# 
# Here the shape `(4,)` means the array is indexed by a single index which runs from 0 to 3. The comma in `(4,)` is there to show that it is a tuple of integers, which is the data type of the shape attribute. Without the comma, it would simply be an integer, not a tuple of integers.
# 
# In most situations the lack of a second dimension is not a problem. If it does turn into a problem (e.g. when you are trying to take a transpose of this vector) you can just call the `reshape` function on the array to generate a new view:

# %%
# Do note the double brackets, as the size is added as a tuple: (rows, columns)
array1 = array1.reshape((4,1))
array1.shape

# %% [markdown]
# For the above examples, we happen to know what we stored in our array, but in some cases we are not aware (like when we imported lots of data). To find out, you can call `dtype`:
# 

# %%
array2.dtype

# %% [markdown]
# ### Initializing regularly used arrays
# There are also basic ways to initialize certain arrays which are used regularly, such as:

# %%
# The empty array (makes an array but doesn't do any initialisation)
print("Ex1: ", np.empty(5))

# Array of 5 floating point zeros
print("Ex2: ", np.zeros(5))

# Array of 5 floating point ones
print("Ex3: ", np.ones(5))

# Array of 5 integer incrementing numbers
print("Ex4: ", np.arange(5))

# Start at 5, stop at 20, do it in steps of 2
print("Ex5: ", np.arange(5, 20, 2))

# Making the identity matrix (ones on the diagonal)
print("Ex6: ")
print(np.eye(5))

# %% [markdown]
# ### Mathematical operations
# 

# %% [markdown]
# The power of NumPy lies in the mathematical operations you can apply on those arrays. These are applied element-wise, which means that the operation is performed on each individual element in the array. Linear Algebra operations, like matrix multiplication or dot product are performed with special NumPy functions, like `np.matmul` or `np.dot`.

# %%
array3 = np.array([[1, 2, 3, 4], [8, 9, 10, 11]])
array3

# %%
# Element-wise multiplication
array3 * array3

# %%
# Element-wise subtraction
array3 - 5

# %%
# Division
1 / array3

# %%
# Raising to a power
arr = np.arange(1, 10, 2)

print(arr)

print(arr ** 3)

print((arr ** 2)[2:5])

# %% [markdown]
# ### You can also apply functions to all elements in an array at once

# %% [markdown]
# The nice thing about NumPy arrays is that it allows you to manipulate the data in arrays without writing explicit loops. For instance look at the addition of all elements in an array:

# %%
# Array of random numbers
a = np.random.rand(65536)

# %%
#calculate the sum of the elements in array
def loopsum(a):
    sum = 0
    for i in range(len(a)):
        sum += a[i]
    return sum

# %%
%timeit loopsum(a)
%timeit np.sum(a)

# %% [markdown]
# You can see that the numpy loop is about 350 times faster than the explicit loop version.
# So be aware in this course to use built-in NumPy tools to manipulate and calculate with arrays.
# Some built-in functions of NumPy can be found [here].
# 
# 
# 
# [here]: https://docs.scipy.org/doc/numpy/reference/ufuncs.html#math-operations
# 
# 

# %% [markdown]
# ### Indexing Arrays

# %%
array4 = np.arange(0, 10)
array4

# %% [markdown]
# There is a minor difference when it comes to indexing compared to Python lists: namely that a NumPy array allows an additional indexing method:

# %%
list3 = [[1, 2, 3], [4, 5, 6]]
array5 = np.array(list3)

# Watch the brackets closely.
print("List: ", list3[1][2])
# Array can use two different approaches
print("Array ", array5[1, 2])
print("Array ", array5[1][2])

# %% [markdown]
# ### Slicing arrays
# Sometimes you do not want the full array, but just parts of it, we can use array slicing for this:

# %%
# Show original array
print(array5)
# We want the element from the second row and third column:
print(array5[1:2,2:3])

# %%
# We can also use it to set the value of multiple entries:
array4 = np.arange(0, 10)
array4[2:5] = 13
print(array4)

# %% [markdown]
# One important thing to note is that a slice is just another _view_ of the underlying data buffer. If you change data in the slice, you are actually changing the data in the underlying data buffer and thus in the orginal array. This is advantageous for the memory efficiency of your program, but sometimes it can cause errors when overlooked.

# %%
array4 = np.arange(0, 10)
# Take a slice, consisting of the 3rd (index 2) to 6th (index 5) element.
slice_array4 = array4[2:6]
# We iterate over all values, setting them to 22
slice_array4[:] = 22
print(slice_array4)
print(array4)

# %% [markdown]
# To prevent this, we can also make a new copy. In this way we do not generate a view, but actually reserve new memory for the object we are making:

# %%
array4 = np.arange(0, 10)
array5 = array4.copy()
print(array4)
print(array5)
array5[:] = 22
print("So did we make a copy?")
print(array4)
print(array5)
print("Seems we did.")

# %% [markdown]
# #### 2D array slicing

# %%
array6 = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
print(array6)
# let's say you only want just the upper right square of 2x2 of the above matrix
# reminder: when indexing with array[start:stop]
# the element at start is *included*, while the elemnt at stop is *excluded*
array6[:2, 1:]

# %% [markdown]
# #### Fancy Indexing
# Sometimes you don't want to retrieve every row, but perhaps skip a few entries. This is easily possible in Python. Let us assume we only want the 2nd, 3rd, 5th, and 7th row in the following example.

# %%
# Below we use a list comprehension (which you should have seen in Introduction to Programming as well)
# To generate an array with 10 rows, and each column goes from 0 to 10.
array7 = np.array([[j for i in range(10)] for j in range(10)])
print(array7)
# As we start at index 0, we actually want the following rows [1, 2, 4, 6].
# Also note the double brackets below.
print("With fancy indexing:")
print(array7[[1, 2, 4, 6]])

# %% [markdown]
# You can do the above even in any order you wish.

# %%
array7[[6, 2, 4, 1]]

# %% [markdown]
# ### Array Transposition

# %%
array8 = np.arange(40).reshape((8, 5))
array8

# %%
# If you want to transpose a matrix you can do this in two ways:

print(np.transpose(array8))
# Or
print(array8.T)

# %% [markdown]
# ### Array Processing
# 
# We can also apply functions on arrays to retrieve specific information from them, or to process the information that is contained. In this section, we will survey a number of these handy functions.

# %% [markdown]
# #### Numpy Where
# 
# When you want to find the location of elements in an array, you can use `np.where`. This function takes in a condition and returns the indices of the array where the condition is true.

# %%
# A simple np.where example:
A = np.array([1, 2, 3, 4])
indices = np.where(A < 3)
print(indices)
print(A[indices])

# %% [markdown]
# #### Numpy Any & All
# 
# As shown before, we can apply a boolean operator on an array, which will return an array with boolean values:

# %%
a = np.arange(9).reshape(3, 3)
bool_arr = a < 4
print(bool_arr)

# %% [markdown]
# Now, what if we want to return each row where any of the elements is true, we can do that with `any` in combination with `where`:

# %%
# Return the indices of the rows where any column (axis=1) is true
indices = np.where(bool_arr.any(axis=1))
bool_arr[indices]
# You can see that the last row is excluded as this row doesn't contain any True values

# %% [markdown]
# And what if we want all elements in the row to be true? We can use `all`:

# %%
# If all values are true, return true (else false)
indices = np.where(bool_arr.all(axis=1))
bool_arr[indices]
# You can see that only the first row is included as this row is the only one to contain only True values

# %% [markdown]
# #### Numpy Unique and `in` checking

# %%
# Sometimes you just want to know all the unique values in a numpy array
# Luckily that function was already implemented for you
letters = ['A', 'B', 'C', 'D', 'D', 'A', 'E', 'F', 'G', 'H', 'Z']
np.unique(letters)

# %%
# We can also easily check whether a big array exists,
# if it exists within a 1D vector.
np.in1d(['X', 'C', 'M', 'Z'], letters)

# %% [markdown]
# ## Step 2: Practical Application of NumPy in Machine Learning

# %% [markdown]
# In this section, we will apply NumPy to a dataset from Scikit-learn. This will demonstrate how NumPy is used in real-world Machine Learning tasks, particularly for data representation and manipulation. 

# %%
import pandas as pd
titanic_data_df= pd.read_csv('data/titanic.csv')
titanic_data = titanic_data_df.to_numpy()


# %% [markdown]
# We can now look at the features present.

# %%
features = titanic_data_df.dtypes
features

# %% [markdown]
# **Question** How many features are present? (Find the length of the list above)

# %%
# START ANSWER
# END ANSWER
print(len(features))

# %% [markdown]
# Usually, before applying any Machine Learning algorithm we always need to inspect the data to get a better idea of its structure and the necessary preprocessing steps that need to be performed.
# 
# **Question** Print the first 10 rows of the data to understand the structure.

# %%
print(titanic_data[:10])

# %% [markdown]
# **Question** What is the shape of `titanic_data`? What does each dimension correspond to?

# %%
print(titanic_data.shape)

# first dimension: rows 

# second dimension: columns (attributes/features)

# %% [markdown]
# **Question** Now, say that from the titanic data we only need the info from the first 15 passengers that tell their id, if they survived and the class in which they were traveling (three first columns). Select the desired subset from the `titanic_array` and put it into the variable `surviving_data`.

# %%
surviving_data = titanic_data[:15, :3]
print(titanic_data.shape)

print(surviving_data)
print('Selection correct: ', surviving_data.shape == (15,3))


# %% [markdown]
# **Question** The 2nd column (index 1) contains the information about the survival status of the passengers. It contains only 0s and 1s. What do these values stand for?

# %% [markdown]
# **Question** From the entire data, can you extract the information of all the passengers that survived? Hint: Use `np.where` to get the indices of the surviving passengers.

# %%
survived_idx = np.where(titanic_data[:, 1] == 1)[0]
titanic_survived = titanic_data[survived_idx]
print(titanic_survived)

# %% [markdown]
# ## Step 3: Practice Numpy
# 

# %% [markdown]
# These are optional exercises, but highly recommended to get you familiar with some basic NumPy operations and tricks. If you get stuck, try to dive into the Numpy documentation first, to see if it can be of help as not all functions you need for the exercises are explained above.

# %% [markdown]
# ### Array Calculations and Array Indexing
# In all exercises below you are not allowed to use a loop.

# %%
# Given two arrays A and B each of the same size calculate their sum (elementwise) and their product (elementwise).
A = np.arange(5)
B = np.arange(5, 10)

def sum_arrays(A, B):
    result = None
    # START ANSWER
    result = A + B
    # END ANSWER
    return result

def multiply_arrays(A, B):
    result = None
    # START ANSWER
    result = A * B
    # END ANSWER
    return result

sum_AB = sum_arrays(A, B)
assert (sum_AB == np.array([ 5,  7,  9, 11, 13])).all()
mult_AB = multiply_arrays(A, B)
assert (mult_AB == np.array([ 0,  6, 14, 24, 36])).all()

sum_AB, mult_AB

# %%
# Given an array A with shape (128,) calculate the mean of the elements at even indexes.
A = np.arange(128)

def mean_even_idx(A):
    result = None
    # START ANSWER
    result = np.mean(A[::2])
    # END ANSWER
    return result

mean_A = mean_even_idx(A)
assert mean_A == 63.0

mean_A

# %%
# Given an array A with shape (N,) make an array with all elements of A in reverse order
# and return as a matrix of size (N, 1).
A = np.arange(6)

def reverse(A):
    result = None
    # START ANSWER
    result = A[::-1].reshape(-1, 1)
    # END ANSWER
    return result

rev = reverse(A)
assert (rev == np.array([[5],[4],[3],[2],[1],[0]])).all()
rev

# %% [markdown]
# #### Two dimensional data arrays
# In this course you will be working a lot with matrices and vectors. The following exercises will let you practice with those.
# 
# Given is data matrix X with shape (m, n)

# %%
m = n = 5
X = np.arange(m * n).reshape(m, n)

print(X)

# %%
# Select the j-th column from the matrix X. What happens if you use X[j]? Is this correct?
j = 3
column = None
# START ANSWER
column = X[:, j]
# X[j] selects row j, not column j.
# END ANSWER

assert (column == np.array([ 3,  8, 13, 18, 23])).all()
column

# %%
# Given an ndarray X with shape (m,n), calculate the mean of each column.
# Try doing this without using np.mean or loops.
# Hint: try to sum up the entries and dividing them by the number of elements

means = 0
# START ANSWER
means = np.sum(X, axis=0) / X.shape[0]
# END ANSWER

assert (means == np.array([10., 11., 12., 13., 14.])).all()
means

# %%
# Now subtract the mean vector you just calculated from all the rows in your matrix leading to the 
# data matrix X_0. Yes this can be done without a loop! Hint: look at array broadcasting.

X_0 = None
# START ANSWER
X_0 = X - means
# END ANSWER

assert (X_0 == np.array([[-10., -10., -10., -10., -10.],
                         [ -5.,  -5.,  -5.,  -5.,  -5.],
                         [  0.,   0.,   0.,   0.,   0.],
                         [  5.,   5.,   5.,   5.,   5.],
                         [ 10.,  10.,  10.,  10.,  10.]])
       ).all()
X_0

# %%
# Given column j, find the largest element and return the entire row of this element.
# Hint: look at the function np.argmax for this.
X = np.random.rand(m, n)
j = 3

# START ANSWER
max_row = X[np.argmax(X[:, j]), :]
assert max_row[j] == np.max(X[:, j])
# END ANSWER

# Please inspect visually whether your code returns the right row
X, max_row

# %% [markdown]
# #### Advanced Linear Algebra

# %% [markdown]
# In Python 3, the @ operator is introduced for matrix multiplication. Let A be an array of shape (m, n) and 
# let B be an array of shape (n, k) then we can write A @ B for the matrix multiplication of A and B. 
# 
# Note that there is conceptual difference between a 1 dimensional array V of size (N,) 
# and a vector V as we know it from linear algebra. In linear algebra, a vector with $N$ elements has dimensions $N \times 1$. 
# A ‘vector’ V as a numpy array has shape (N,).
# 
# 
# 

# %%
# Calculate the inner product of two vector v and w both of shape (N,).
# Validate your result by computing the dot product using multiply and sum operations.
v = np.arange(5)
w = np.arange(5, 10)

# START ANSWER
inner_product = v @ w
assert inner_product == np.sum(v * w)
inner_product
# END ANSWER

# %%
# Calculate the product of a matrix A of shape (M,N) with a vector v of shape (N,).
m = n = 5
X = np.arange(m * n).reshape(m, n)
v = np.arange(n)

product = None
# START ANSWER
product = X @ v
# END ANSWER

assert (product == np.array([ 30,  80, 130, 180, 230])).all()
product        


