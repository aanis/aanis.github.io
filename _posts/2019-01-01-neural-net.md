---
title: "A dummies guide to neural net"
layout: post
date: 2019-01-01
tags: neural net, dummy, guide
---


![](https://cdn-images-1.medium.com/max/800/1*vohhSJ0H0vn976rAHVEccg.png)

A neural net is simply *a network of neurons.*

Inspired by a neurons in the human brain; a single layer neural net is called a
perceptron. A perceptron performs a simple linear calculation for binary
classification.

A perceptron is a type of a feed forward neural net which only does a forward
pass. A forward pass is when an output is generated from an input via a simple
linear function.

In neural nets if we stack these perceptrons upon each other a mesh of
perceptrons are formed. This mesh of perceptron is known as a multi-layer
perceptron (MLP). We use MLP for multi-class classification problems. MLP use
backpropogation to solve problems.

*****

### **Example of a feed forward neural net**

![](https://cdn-images-1.medium.com/max/800/1*zd2AuD53JxsJCBeyelb-2A.png)

*****

### **Diagram solution**

The input of the perceptron is a matrix of numbers which repreesent a binary
class.

If we look at the first neuron x0 we see the input is multiplied by a particular
weight.

We multiply each input with the assigned weight.

    x = input, w = weight 

    input + weight = E

    (x0 * w0) + (x1 * w1) + (x2 * w2)

    (2.0 * 0.1) + (3.0 * 0.5) + (4.0 * 0.9) = 5.3

After we multiply each input by the weight we are left with the value of **E**
which is 5.3

    b = bias, Y = b + E

    E + bias = Y

    5.3 + (-2.0) =  3.3

After that we simply add the bias which is -2* *and we are left with the value**
**of **Y **which is *3.3*

This value (**Y)** is then passed through an activation function.

*****

### **Why do we use an activation function?**

Our **Y **value has no bounds it could be infinite. Hence, we need to pass it
through an activation function to give it a restricted finite value so our
neural net can make a prediction.

    A = activation function 

    A = sigmoid(Y)

In this case we use a sigmoid non-linear function.

**Y** is then passed through the activation function of your choosing (sigmoid,
tanh, relu) which converts our **Y** of 3.3 into **A** of 0.96.

Now we can make a binary classification depending on the threshold of the the
value we calculated.


