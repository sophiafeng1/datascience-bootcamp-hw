#!/usr/bin/env python
# coding: utf-8

# # The Basics of NumPy and Pandas

# **Numpy** is the core library for scientific computing in Python. It provides a high-performance multidimensional **array object**, and tools for working with these arrays. If you are already familiar with MATLAB, you might find this [tutorial](http://wiki.scipy.org/NumPy_for_Matlab_Users) useful to get started with Numpy.

# ###Why use Numpy Arrays instead of Lists?
# 

# *   Data types: Arrays in NumPy are **homogeneous**, meaning all elements must be of the same data type (e.g., integers, floats), whereas lists can contain elements of different data types.
# 
# *   Memory efficiency: Arrays are more **memory efficient** compared to lists because they store data in a contiguous block of memory. Lists, on the other hand, store references to objects in memory, which can result in more memory overhead.
# 
# *   Performance: Operations on arrays are generally faster and more efficient than operations on lists, especially for large datasets, due to **NumPy's implementation in C** and optimized algorithms.
# 
# *   Functionality: NumPy arrays come with a wide range of **built-in functions and methods** for mathematical operations, linear algebra, statistical analysis, and more. Lists have a more limited set of built-in functions and methods.
# 
# 
# 

# To use Numpy, we first need to import the numpy package. By convention, we import it using the alias np. Then, when we want to use modules or methods in this library, we preface them with np.
# 
# 

# In[2]:


import numpy as np


# ### Arrays and array construction

# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

# We can create a `numpy` array by passing a Python list to `np.array()`.

# In[3]:


a = np.array([1, 2, 3])  # Create a rank 1 array


# This creates the array we can see on the right here:
# 
# ![](http://jalammar.github.io/images/numpy/create-numpy-array-1.png)

# In[4]:


print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5                 # Change an element of the array
print(a)


# To create a `numpy` array with more dimensions, we can pass nested lists, like this:
# 
# ![](http://jalammar.github.io/images/numpy/numpy-array-create-2d.png)
# 
# ![](http://jalammar.github.io/images/numpy/numpy-3d-array.png)

# In[5]:


b = np.array([[1,2],[3,4]])   # Create a rank 2 array  1，2 first row, 3,4 second row; 2 columns
print(b)
print(b.shape)   #2行2列，第一个2是代表行


# In[7]:


c=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(c)
print(c.shape)


# There are often cases when we want numpy to initialize the values of the array for us. numpy provides methods like `ones()`, `zeros()`, and `random.random()` for these cases. We just pass them the number of elements we want it to generate:
# 
# ![](http://jalammar.github.io/images/numpy/create-numpy-array-ones-zeros-random.png)

# Sometimes, we need an array of a specific shape with “placeholder” values that we plan to fill in with the result of a computation. The `zeros` or `ones` functions are handy for this:

# In[16]:


a = np.ones(3)  # Create an array of all ones   一行3个数   放一个数字，就生成一行多列
print(a)
b=np.random.random(3)
print(b)


# We can also use these methods to produce multi-dimensional arrays, as long as we pass them a tuple describing the dimensions of the matrix we want to create:
# 
# ![](http://jalammar.github.io/images/numpy/numpy-matrix-ones-zeros-random.png)
# 
# 

# In[17]:


a = np.zeros((2,2))  # Create an array of all zeros
print(a)
b = np.ones((1,2))   # Create an array of all ones   1行2列
print(b)
c = np.random.random((2,2)) # Create an array filled with random values
print(c)


# Numpy also has two useful functions for creating sequences of numbers: `arange` and `linspace`.
# 
# The `arange` function accepts three arguments, which define the start value, stop value of a half-open interval, and step size. (The default step size, if not explicitly specified, is 1; the default start value, if not explicitly specified, is 0.)
# 
# The `linspace` function is similar, but we can specify the number of values instead of the step size, and it will create a sequence of evenly spaced values.

# In[5]:


f = np.arange(10,50,5)   # Create an array of values starting at 10 in increments of 5
print(f)


# Note this ends on 45, not 50 (does not include the top end of the interval).

# In[18]:


g = np.linspace(0., 1., num=5)   #几等分
print(g)


# In[3]:


# Using linspace
linspace_array = np.linspace(0, 10, 5)
print("Linspace Array:", linspace_array)

# Using arange
arange_array = np.arange(0, 10, 2)
print("Arange Array:", arange_array)


# In[4]:


print(np.arange(1,10))    #一行 每个数字中间没有中括号
print(np.array([[1],[1],[1],[1]]))   #数字中间有中括号，则是列，4列


# Sometimes, we may want to construct an array from existing arrays by “stacking” the existing arrays, either vertically or horizontally. We can use `vstack()` (or `row_stack`) and `hstack()` (or `column_stack`), respectively.

# In[19]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack((a,b))


# In[20]:


a = np.array([[7], [8], [9]])
b = np.array([[4], [5], [6]])
np.hstack((a,b))


# ## NumPy Array Attributes

# In[5]:


x1=np.array([5,6,7,8,9])
print(x1.shape)
print(x1)


# Each array has attributes ``ndim`` (the number of dimensions), ``shape`` (the size of each dimension), and ``size`` (the total size of the array):

# In[10]:


print("x1 ndim: ", x1.ndim)
print("x1 shape:", x1.shape)
x2 = np.random.random((3, 4))  # Two-dimensional array 2个维度：行和列
print("x2 size: ", x2.size)
print(x2.ndim)
print(x2.shape)
print(np.random.random((3, 4)))     #3行4列 


# ####Datatypes

# Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

# In[11]:


x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)


# You can read all about numpy datatypes in the [documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

# ## Array Indexing: Accessing Single Elements

# If you are familiar with Python's standard list indexing, indexing in NumPy will feel quite familiar.
# In a one-dimensional array, the $i^{th}$ value (counting from zero) can be accessed by specifying the desired index in square brackets, just as with Python lists:

# In[28]:


x1


# In[29]:


x1[0]


# In[30]:


x1[4]


# To index from the end of the array, you can use negative indices:

# In[31]:


x1[-1]


# In[32]:


x1[-2]


# In a multi-dimensional array, items can be accessed using a comma-separated tuple of indices:

# In[12]:


x2


# In[13]:


x2[0, 0]


# In[14]:


x2[2, 0]    # 第0行，第一行，第二行  2就是第三行，0就是第一个


# In[15]:


x2[2, -1]    #第三行，最后一个


# Values can also be modified using any of the above index notation:

# In[68]:


x2[0, 0] = 12
x2


# Keep in mind that, unlike Python lists, NumPy arrays have a fixed type.
# This means, for example, that if you attempt to insert a floating-point value to an integer array, the value will be silently truncated. Don't be caught unaware by this behavior!

# In[16]:


print(x1)
x1[0] = 3.14159  # this will be truncated!
x1


# ## Array Slicing: Accessing Subarrays

# Just as we can use square brackets to access individual array elements, we can also use them to access subarrays with the *slice* notation, marked by the colon (``:``) character.
# The NumPy slicing syntax follows that of the standard Python list; to access a slice of an array ``x``, use this:
# ``` python
# x[start:stop:step]
# ```
# If any of these are unspecified, they default to the values ``start=0``, ``stop=``*``size of dimension``*, ``step=1``.
# We'll take a look at accessing sub-arrays in one dimension and in multiple dimensions.

# ### One-dimensional subarrays

# In[17]:


x = np.arange(20)
x


# In[18]:


x[:5]  # first five elements


# In[19]:


x[5:]  # elements after index 5


# In[20]:


x[4:7]  # middle sub-array


# In[21]:


x[::2]  # every other element  # What if I want odd numbers?


# A potentially confusing case is when the ``step`` value is negative.
# In this case, the defaults for ``start`` and ``stop`` are swapped.
# This becomes a convenient way to reverse an array:

# In[46]:


x[::-1]  # all elements, reversed


# ### Multi-dimensional subarrays
# 
# Multi-dimensional slices work in the same way, with multiple slices separated by commas.
# For example:

# In[22]:


x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array  
x2


# In[23]:


x2[:2, :3]  # two rows, three columns  the first two rows, the first three columns 


# In[24]:


x2[:3, ::2]  # all rows, every other column


# Finally, subarray dimensions can even be reversed together:

# In[25]:


x2[::-1, ::-1]


# #### Accessing array rows and columns
# 
# One commonly needed routine is accessing of single rows or columns of an array.
# This can be done by combining indexing and slicing, using an empty slice marked by a single colon (``:``):

# In[26]:


print(x2[:, 0])  # first column of x2    #全部的行，第一列


# In[27]:


print(x2[0, :])  # first row of x2       #全部的列，第一行


# In the case of row access, the empty slice can be omitted for a more compact syntax:

# In[28]:


print(x2[0])  # equivalent to x2[0, :]


# ### Subarrays as no-copy views
# 
# One important–and extremely useful–thing to know about array slices is that they return *views* rather than *copies* of the array data.
# This is one area in which NumPy array slicing differs from Python list slicing: in lists, slices will be copies.
# Consider our two-dimensional array from before:

# In[69]:


print(x2)


# Let's extract a $2 \times 2$ subarray from this:

# In[70]:


x2_sub = x2[:2, :2]     #第0行到第二行（但不包括第二行），第0列到第2列（但不包括第二列）
print(x2_sub)     


# Now if we modify this subarray, we'll see that the original array is changed! Observe:

# In[71]:


x2_sub[0, 0] = 99
print(x2_sub)


# In[72]:


print(x2)     


# This default behavior is actually quite useful: it means that when we work with large datasets, we can access and process pieces of these datasets without the need to copy the underlying data buffer.

# ### Creating copies of arrays
# 
# Despite the nice features of array views, it is sometimes useful to instead explicitly copy the data within an array or a subarray. This can be most easily done with the ``copy()`` method:

# In[73]:


x2_sub_copy = x2[:2, :2].copy()     #新建了一个矩阵，可以避免在原矩阵上操作
print(x2_sub_copy)


# If we now modify this subarray, the original array is not touched:

# In[10]:


x2_sub_copy[0, 0] = 42
print(x2_sub_copy)


# In[11]:


print(x2)


# ## Reshaping of Arrays
# 
# Another useful type of operation is reshaping of arrays.
# The most flexible way of doing this is with the ``reshape`` method.
# For example, if you want to put the numbers 1 through 9 in a $3 \times 3$ grid, you can do the following:

# In[14]:


grid = np.arange(1, 10).reshape((3,3))   #reshape是用来改变形状的
print(grid)


# In[15]:


grid = np.arange(1,10)
print(grid)
grid = grid.reshape((3,3))
print(grid)


# Note that for this to work, the size of the initial array must match the size of the reshaped array.
# 
# Another common reshaping pattern is the conversion of a one-dimensional array into a two-dimensional row or column matrix.
# This can be done with the ``reshape`` method, or more easily done by making use of the ``newaxis`` keyword within a slice operation:

# In[19]:


x = np.array([1, 2, 3])
# row vector via reshape
print(x.shape)
print(x)
x = x.reshape((3,1))
print(x.shape)
print(x)


# In[31]:


x = np.array([1, 2, 3])   #？？？？
print(x.shape)
# row vector via newaxis
x = x[np.newaxis, :]   #It adds an axis of length 1 to the array, effectively increasing its dimensionality.
print(x)
print(x.shape)    #从一阶变为2阶 但没有给实际的数，只是格式变了


# In[25]:


x = np.array([1, 2, 3])
# column vector via reshape
x.reshape((3, 1))


# In[29]:


# column vector via newaxis
x = np.array([1, 2, 3])
x=x[:, np.newaxis]
print(x.shape)


# You will come across such transformations throughout any data science problem.
# 

# ## Array Concatenation and Splitting
# 
# All of the preceding routines worked on single arrays. It's also possible to combine multiple arrays into one, and to conversely split a single array into multiple arrays. We'll take a look at those operations here.

# ### Concatenation of arrays
# 
# Concatenation, or joining of two arrays in NumPy, is primarily accomplished using the routines ``np.concatenate``, ``np.vstack``, and ``np.hstack``.
# ``np.concatenate`` takes a tuple or list of arrays as its first argument, as we can see here:

# In[ ]:


x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])


# You can also concatenate more than two arrays at once:

# In[ ]:


z = [99, 99, 99]
print(np.concatenate([x, y, z]))


# It can also be used for two-dimensional arrays:

# In[75]:


grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(grid.shape)


# In[76]:


# concatenate along the first axis    竖着贴变成新的矩阵
new_grid = np.concatenate([grid, grid], axis=0)
new_grid.shape
print(new_grid)


# In[77]:


# concatenate along the second axis (zero-indexed)  横着贴变成新的矩阵
new_grid_col = np.concatenate([grid, grid], axis=1)
new_grid_col.shape
print(new_grid_col)


# Numpy also provides many useful functions for performing computations on arrays, such as `min()`, `max()`, `sum()`, and others:
# 
# ![](http://jalammar.github.io/images/numpy/numpy-matrix-aggregation-1.png)

# In[35]:


data = np.array([[1, 2], [3, 4], [5, 6]])

print(np.max(data))  # Compute max of all elements; prints "6"
print(np.min(data))  # Compute min of all elements; prints "1"
print(np.sum(data))  # Compute sum of all elements; prints "21"


# Not only can we aggregate all the values in a matrix using these functions, but we can also aggregate across the rows or columns by using the `axis` parameter:
# 
# ![](http://jalammar.github.io/images/numpy/numpy-matrix-aggregation-4.png)

# In[36]:


data = np.array([[1, 2], [5, 3], [4, 6]])    #???

print(np.max(data, axis=0))  # Compute max of each column; prints "[5 6]"  纵轴找最大值，有两列所以会挑出2个
print(np.max(data, axis=1))  # Compute max of each row; prints "[2 5 6]"


# Let's look at a practical example using the numpy attributes we just discussed 💻

# ## Practice Questions for numpy
# 1. Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.
# 2. Find common elements between A and B. [Hint : Intersection of two sets]
# 3. Extract all numbers from A which are within a specific range. eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]
# 4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
# ```
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
# ```

# In[ ]:


## Optional Practice Question

#Find the mean of a numeric column grouped by a categorical column in a 2D numpy array

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


numeric_column = iris[:, 1].astype('float')  # sepalwidth
grouping_column = iris[:, 4]  # species

output = []
"""Your code goes here"""

output


# In[ ]:





# ## Starting with Pandas
# 
# Pandas is a powerful and popular open-source Python library used for data manipulation (cleaning, filtering, sorting, reshaping, restructuring, aggregating, joining) and analysis. It provides data structures and functions designed to make working with structured (tabular) data easy and intuitive.
# 
# Check out the [documentation](https://pandas.pydata.org/docs/reference/index.html) as you code.

# In[32]:


#we start from the very basics...import!

import pandas as pd
import numpy as np


# The **DataFrame** is a two-dimensional labeled data structure with columns of potentially different types, similar to a spreadsheet or SQL table. It provides powerful indexing, slicing, and reshaping capabilities, making it easy to manipulate and analyze data.

# ## PART 1: Getting and Knowing your Data
# 

# In[33]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'

chipo = pd.read_csv(url, sep='\t')  #??? \t?


# ### See the first 10 entries

# In[34]:


chipo.head(10)  #Returns the first 10 rows    head()  是默认看前5行


# ### Print the last elements of the data set.

# In[35]:


chipo.tail()   #Returns the last 5 rows


# ### What is the number of observations in the dataset?

# In[5]:


chipo.shape    #Return a tuple representing the dimensionality of the DataFrame.   #How to access no of rows and columns?


# ### Another way

# In[69]:


chipo.info()


# ### What is the number of columns in the dataset?

# In[78]:


chipo.shape[1]   
a=chipo.shape
print(a)
print(a[0],a[1])


# ### What are the different columns in our dataset?

# In[44]:


chipo.columns


# In[71]:


chipo.index


# ### How many items were orderd in total?

# In[72]:


total_items_orders = chipo.quantity.sum()
total_items_orders


# ### Check the item price type

# In[79]:


chipo.item_price.dtype     #???
# It is a python object


# How much was the revenue for the period in the dataset?

# In[83]:


#chipo['item_price'] = chipo['item_price'].str[1:] #.str[1:]是对item_price列中的每个元素执行的一个字符串操作。这个操作的目的是去除每个价格值前面的第一个字符。通常，这样做是因为价格值前面有一个货币符号（如美元符号$），我们想要去除这个符号，只保留数字部分，以便后续进行数值运算。
chipo['item_price'] = chipo['item_price'].astype('str')
for i in range(chipo['item_price'].size):
    chipo.iloc[i,4] = chipo.iloc[i,4][1:]   #读取第i行第四列反回来的字符串，然后只保留第1个到最后一个，不要第0个
chipo['item_price'] = pd.to_numeric(chipo['item_price'])  #原先是object，把oject转换成数字类型


# In[84]:


revenue = (chipo['quantity']* chipo['item_price'])    #???
revenue = revenue.sum()

print('Revenue was: $' + str(np.round(revenue,2)))


# ### How many orders were made in the period?

# In[39]:


orders = chipo.order_id.value_counts().count()         # value_counts: returns the frequency count of unique values in the 'order_id' column of the DataFrame chipo
orders    
#.value_counts()是一个方法，用于计算order_id列中每个不同值出现的次数。这意味着它会告诉我们每个订单ID在数据集中出现了多少次，从而了解每个订单的频率。
.count()是在value_counts()方法的结果上调用的另一个方法。value_counts()返回一个新的Series，其中索引是order_id的唯一值，值是这些唯一值出现的次数。在这个Series上调用.count()方法会返回这个Series的长度，即order_id的唯一值的数量。


# ### How many different items are sold?
# 

# In[40]:


chipo.item_name.value_counts().count()   


# ## PART B: Filtering and Sorting Data

# ### What is the price of each item?
# 

# In[41]:


chipo[(chipo['item_name'] == 'Chicken Bowl') & (chipo['quantity'] == 1)]


# ### Sort by the name of the item

# In[ ]:


chipo.item_name.sort_values()     


# ### OR

# In[ ]:


chipo.sort_values(by = "item_name")


# ### What was the quantity of the most expensive item ordered?

# In[86]:


chipo.sort_values(by = "item_price", ascending = False).head(1)    
#chipo.sort_values(by = "item_price", ascending = True).head(1)


# ### How many times was a Veggie Salad Bowl ordered?

# In[ ]:


chipo_salad = chipo[chipo.item_name == "Veggie Salad Bowl"]
len(chipo_salad)


# ### Trying some different dataset

# In[ ]:


drinks = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv')
drinks.head()


# ### Which continent drinks more beer on average?

# In[ ]:


drinks.groupby('continent').beer_servings.mean()  #???


# ### For each continent print the statistics for wine consumption.

# In[ ]:


drinks.groupby('continent').wine_servings.describe()


# ### Print the median alcohol consumption per continent for every column

# In[ ]:


drinks.groupby('continent').mean()


# ### Print the median alcohol consumption per continent for every column
# 

# In[ ]:


drinks.groupby('continent').median()


# ### Print the mean, min and max values for spirit consumption.

# In[ ]:


drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])  #？？？


# ### Trying some more different functionalities

# In[45]:


csv_url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv'
df = pd.read_csv(csv_url)
#df
stud_alcoh = df.loc[: , "school":"guardian"]
stud_alcoh.head()


# In[46]:


capitalizer = lambda q: q.capitalize()  #A lambda function in Python is a small anonymous function that can have any number of arguments, but can only have one expression.
                                        #They are defined using the lambda keyword, followed by a list of arguments, a colon, and then the expression to be evaluated.
 #    冒号后面是返回值              # Lambda functions are often used when you need a simple function for a short period of time.


# In[ ]:


def capttalizer(q):
    return q.capitalize()


# In[47]:


stud_alcoh['Mjob'].apply(capitalizer)    #没有本地操作，这个不会改变函数本身
stud_alcoh['Fjob'].apply(capitalizer)
stud_alcoh.tail()


# In[112]:


stud_alcoh['Mjob'] = stud_alcoh['Mjob'].apply(capitalizer)  #这个才是改变数值本身
stud_alcoh['Fjob'] = stud_alcoh['Fjob'].apply(capitalizer)
stud_alcoh.tail()


# ### Here instead of just using the existing the data, we will create our own dataframe/dataseries

# **pd.Series** is a one-dimensional labeled array-like data structure in pandas. It can hold data of any type (integers, floats, strings, etc.) and is similar to a **one-dimensional NumPy array** or a Python list. However, unlike a NumPy array, a pd.Series can have *custom row labels*, which are referred to as the index.

# In[6]:


a = pd.Series([1,2,3])
a


# You can specify custom index labels for a pandas Series by passing a list of index labels to the index parameter when creating the Series.

# In[92]:


data = [10, 20, 30]
custom_index = ['A', 'B', 'C']
s = pd.Series(data)#index=custom_index)    #放的位置固定数值在前，index在后 会有固定的index
s


# In[93]:


a = pd.Series(data,index=custom_index)   #自己设定index
a


# In[50]:


print('Data passed as a list')
df_list = pd.DataFrame([['May1', 32], ['May2', 35], ['May3', 40], ['May4', 50]])
print(df_list)


# In[ ]:





# In[51]:


print('Data passed as dictionary')
df_dict = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]},dtype = float)
print(df_dict)


# In[52]:


# Rename columns

df_dict.rename(columns={'A': 'a'})

# inplace by default is false
# if inplace = True is not set then the changes are not made on the original df but only a temp df is made with changes
df_dict


# In[53]:


# changes made for original df

df_dict.rename(columns={'A': 'a'}, inplace=True)
df_dict


# In[54]:


# Reset column names
# Tip: remember to pass the entire list in this case

df_dict.columns = ['a', 'b']
df_dict.head()


# In[55]:


# Defining columns, index during dataframe creation

df_temp = pd.DataFrame([['October 1', 67], ['October 2', 72], ['October 3', 58], ['October 4', 69], ['October 5', 77]], index = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'], columns = ['Month', 'Temperature'])
df_temp
#???


# ## Practice Questions for Pandas
# 
# 1. From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).
# 
# ```
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# ```
# 
# 2. Replace missing values in Min.Price and Max.Price columns with their respective mean.
# 
# ```
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# ```
# 
# 3. How to get the rows of a dataframe with row sum > 100?
# 
# ```
# df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
# ```

# In[ ]:





# In[ ]:


#homework


# In[56]:


# 1. From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df


# In[61]:


df1=df.loc[::20,"Manufacturer":"Type"]   #读取全部行，但是输出每20间隔
df1


# In[99]:


#q2 Replace missing values in Min.Price and Max.Price columns with their respective mean (check documentation).
#df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
#df.info()
df['Min.Price'].fillna(np.nanmean(df['Min.Price']),inplace=True)
df['Max.Price'].fillna(np.nanmean(df['Max.Price']),inplace=True)
#a1=np.mean(df['Min.Price'].isnull)
#a2 = np.mean(df['Max.Price'].isnull)
#print('There are {a1} Nan in min.Price and {a2} Nan in max.price'.format(a1=a1,a2=a2))


# In[100]:


df.info()


# In[108]:


#q3 How to get the rows of a dataframe with row sum > 100?
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
print(df)
rows= []
for i in range(df.shape[0]):
    if np.sum(df.iloc[i,:])>100:
        rows.append(i)
print(rows)


# In[ ]:





# In[110]:


#q4 Create a 4x4 NumPy array filled with random integers between 1 and 100. Then, reshape this array into two separate 2D arrays, where one represents the rows and the other represents the columns. Write a function, preferably using a lambda function, to calculate the sum of each row and each column separately, and return the results as two separate NumPy arrays
m=np.random.randint(1,101,size=(4,4))
print(m)
row_ = lambda x: np.sum(x,axis=1)
col_ = lambda x: np.sum(x,axis=0)
row_sum = row_(m)
col_sum = col_(m)
print(row_sum)
print(col_sum)


# In[ ]:




