---
title: "A dummies guide to the math of AI"
layout: post
date: 2019-01-01
tags: math, ai, guide, dummy
---



Before you begin your journey in Machine Learning(ML) and Deep Learning (DL) you
should at least know the basic math behind what you will be doing. Although you
can still dive right into algorithms and implementation of code you will have a
severe handicap against your data science peers.

Everyone is at some level daunted by math. But once you understand its real
world application to data science, it becomes inherently more fun and useful.

![](https://cdn-images-1.medium.com/max/800/1*FPP6LhjFdFriQFzLv0gdDg.gif)

The mathematics for machine learning can be divided into 3 main categories:

1.  **Linear Algebra**
1.  **Calculus**
1.  **Statistics and Probability**

*****

### Linear Algebra

**Question: Why study linear algebra?**

**Answer:** By using linear algebra we can solve [“linear
equations”](https://en.wikipedia.org/wiki/Linear_equation). Linear equations can
be represented in the forms of matrices. A matrix in machine learning is how we
represent our features. We can represent data in 0 dimension (scalar), 1
dimension (vector), 2 dimensions (matrix) and n-dimensions (tensor).

    A simple linear equation is y = wx + b

*where;*

    y = y-axis, x = x-axis, w = slope, b = y-intercept

*However in machine learning;*

    y = prediction, b= bias, w = weight of feature, x = feature

**Question: Why do we need to study logarithms?**

**Answer:** In machine learning you will have to deal with big data (millions of
rows and countless features). Logs help us express large numbers efficiently.
And most importantly they will help us
[solve](https://www.khanacademy.org/math/algebra2/exponential-and-logarithmic-functions/solving-exponential-equations-with-logarithms/a/solving-exponential-equations-with-logarithms)
exponential equations like sigmoid (know as activation function in deep
learning). After studying linear algebra you will learn how to solve linear
equations, represent your data in dimensions and understand logarithms.

### Tutorials

**For beginners (high school math only): Start with this** [tutorial.](https://www.khanacademy.org/math/algebra/introduction-to-algebra)

**For intermediate (college level math only): Start with this** [tutorial.](https://www.khanacademy.org/math/linear-algebra)

> Fun fact: The world algebra comes from [Muḥammad ibn Mūsā al-Khwārizmī
> ](https://en.wikipedia.org/wiki/Muá¸¥ammad_ibn_MÅ«sÄ_al-KhwÄrizmÄ«)(ca.
750–ca. 850) who used it to solve linear and quadratic equations.

*****

### **Calculus**

![](https://cdn-images-1.medium.com/max/800/1*PIGwsqIAXbUs5vcVLF_FLg.jpeg)

Calculus tell us how things change. In machine learning we make predictions and
calculus helps us make them. In calculus you will need to study derivatives,
partial derivatives, gradients and chain rules.

**Question: Why study derivatives?**

**Answer:** A derivative simply shows the rate of change; the amount by which a
function is changing at one given point. In machine learning we will need to
find the minimum point in our function where the prediction we make is the
optimal. Derivatives help us do that. Partial derivatives are a very similar to
derivatives as well.

**Question: Why study gradients?**

**Answer:** A gradient is simply the slope of a graph. In machine learning we
use a powerful optimization technique called gradient descent known as
backpropagation in deep learning. Gradient descent is simple terms help us to
find the [local
minima](https://upload.wikimedia.org/wikipedia/commons/6/68/Extrema_example_original.svg)
which reduces our prediction error.

**Great visual tutorial on backpropagation can be found** [here.](https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/)

### Tutorials

**For beginners (high school math only): Start with this** [tutorial](https://www.khanacademy.org/math/precalculus)

**For intermediate (college level math only): Start with this** [tutorial](https://www.khanacademy.org/math/ap-calculus-ab/ab-derivative-intro)

> Fun fact: [Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton) and
> [Gottfried Leibniz](https://en.wikipedia.org/wiki/Gottfried_Leibniz)
independently discovered calculus in the mid-17th century. Both died disputing
who came up with it first.

*****

### **Statistic and Probability**

In machine learning in any type of problem either regression on classification
your algorithm will compute a probability from the features. In order to
interpret what the accuracy means we will need to study stats and probability.

![](https://cdn-images-1.medium.com/max/800/1*Uny-ym1ashJxGKzn1Gmwhw.png)

**Question: Why study statistics?**

**Answer**: Stats is a powerful tool for data scientists; you will learn how to
analyze data and visualize data. Stats is mainly used in the data preprocessing
stage.

**Types of variables:**

**Discrete variables:** Variables which can be counted *(e.g. number of lions)*

**Continuous variables:** Variables which can be measured *(e.g. height,weight)*

**Question: Why use summary statistics?**

**Answer**: So we can quickly summarize the most important points of our data.
Summary statistics includes; central tendency, mean, median, mode, standard
deviation, skewness, kurtosis, range, interquartile range and charts (histogram,
scatter plot, pie chart, line chart etc.)

**Central Tendency:** Describes the central tendency of a data via mean,
median, mode

**Mean:** Sum of all observations/ number of observations

**Median:** The middle observation

**Mode**: The most common observation

**Range:** All respective observations in a group

**Interquartile Range**: Range of observations largest to smallest

**Variance:** Squared difference of observation from mean / number of
observation

**Standard deviation:** Square root of variance.

**Question:** Why is Standard deviaton so important?

**Answer:** We use standard deviation to measure how our data is distributed.
The greater the spread the greater the standard deviation.

**Hypothesis testing:** In data science, you always start with a hypothesis.
Your goal is to reject the null hypothesis.

**Types of Error 1&2:**

![](https://cdn-images-1.medium.com/max/800/1*OGxnsYE_4ZETbEHTxenVOA.jpeg)
<span class="figcaption_hack">Type 2 errors are more dangerous than Type 1 errors</span>

**Skewness:** Measure the lack of symmetry in our data. Symeetrical data will
have perfect symmetry on both sides.

**Kurtosis:** Measure whether the data are heavy-tailed or light-tailed relative
to a normal distribution.

**Normal distribution:** Values plotted on a graph which are bell shaped. Great
primer on normal distribution and its importance can be found
[here](https://www.mathsisfun.com/data/standard-normal-distribution.html)**.**

*If data looks normal use = z, t, ANOVA, Chi, F test*

*If data is skewed use = Chi, F test*

**Question: Why use Probability?**

**Answer:** Probability simply means chance of an event happening. We use the
range of 0 – 100% to describe the chance of a particular event happening. 0
being no chance, 100 being an absolute.

**Conditional Probability:** Simply means the chance of an A event happening, if
event B has already happened.

### Tutorials

**For beginners (high school math only): Start with this** [tutorial.](https://www.khanacademy.org/math/statistics-probability/analyzing-categorical-data/one-categorical-variable/v/identifying-individuals-variables-and-categorical-variables-in-a-data-set)

**For intermediate (college level math only): Start with this** [tutorial.](https://www.khanacademy.org/math/statistics-probability)

> Fun Fact: [Al-Kindi](https://en.wikipedia.org/wiki/Al-Kindi) developed the first
> code breaking algorithm based on [frequency
analysis](https://en.wikipedia.org/wiki/Frequency_analysis).
