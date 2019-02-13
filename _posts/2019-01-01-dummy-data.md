---
title: "A dummies guide to data normalization"
layout: post
date: 2019-01-01
tags: announcement
---


Deploying a neural network is an arduous process. One of the most important
stages in developing a neural net is to first normalize the data. In this guide,
I will explain **why is normalization important,and finally how to
normalize your data.**

*****

**Problem:** To predict if a person has children given the feature provided.

The input features (x-values, independent variables) need to be first normalized
before we feed it to the neural network. In machine learning we call this
process *feature scaling* or *data preprocessing.*

![](https://cdn-images-1.medium.com/max/800/1*IV63jT3z4aJOGyHjEMdiGg.png)

**Given the example above we have the following features:**

**Input features** (x-values or independent values): **Age, Income, Sex,
Education, Marital Status and Religion**

**Output feature** (y-value or dependent value): **Children**

*****

### **Question: Why is normalization important?**

**Answer:** We have to normalize our data because our features do not have a
uniform scale.

Most, if not all classifiers in machine learning calculate the Euclidean
distance between the features. Euclidean distance is the “ordinary”
straight-line distance between two points (vectors in a neural net) in Euclidean
space.

Euclidean space is simply a 2 or 3 dimensional space. Hence, we normalize our
features to remove any bias in our model. Also normalized data converges faster
during backpropogation.

![](https://cdn-images-1.medium.com/max/800/1*PQ7T05W7qZU8fxLQbdkOSA.png)
<span class="figcaption_hack"></span>

*****

### **Question: How to normalize your data**

**Answer: **In order to normalize your data, you will first need to learn the
following methods:

1.  **Mix-max normalization**

![](https://cdn-images-1.medium.com/max/800/1*m-c4ARwLehrvsBW84w8jcA.png)

Take a value, subtract it by the minimum value and divide it by the difference
of the maximum and minimum value. Normalizes the range of features to the range
[0, 1] or [−1, 1].






**2. Z-score normalization**

![](https://cdn-images-1.medium.com/max/800/1*4pGbXvZ_kUZB1bZZIimNaw.png)

Take a value, subtract it by the mean of all values and divide it by the
standard deviation of all the values. After normalization mean will be 0 and
standard deviation will be 1.






**3.** **Constant Normalization**

Take your value and divide by a constant. Rule of thumb is to use constant
values of a multiple of 10.


     = 55

    = 10

     = 55/10 = 5.5

**4.** **Binary Encoding**

Categorical values like Gender can be encoded in either 0 or 1. Male can be
encoded to 1, Female to 0.

Categorical values like Gender can also be encoded in either -1 or 1. Males can
be encoded to -1, Female to 1.

**5. Manhattan Encoding**

If we have non-binary categorical data we can used Manhattan encoding which uses
0 or 1 o indicate if feature is included or excluded.

*For e.g. In Religion class we can encode:*

*Muslim as a scalar of* **[ 1 0 0 ]**

*Hindu as a scalar* ***[ 0 1 0 ]**

*Christian as a scalar* **[ 0 0 1 ]**

> Pro Tip: Use
> [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
and
[OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
for feature scaling in sci-kit learn library when coding

