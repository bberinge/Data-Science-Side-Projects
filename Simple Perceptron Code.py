#!/usr/bin/env python
# coding: utf-8

# # Coding a Single Perceptron 
# This file illustrates a single perceptron with two input variables that uses the sign function as its activation function.  It is demonstrated that such a perceptron can represent every boolean function except for XOR() and XNOR().     

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


def perceptron(inputs, weights): #computes the output of a single perceptron
    linear_sum = 0
    for i in range(0, len(inputs)):
        linear_sum += inputs[i]*weights[i]
    if linear_sum > 0:
        sign = 1
    else:
        sign = -1
    return sign


# In[4]:


def update_weights(weights, inputs, target, output, learning_rate): #updates weights for a single perceptron for a single training example
    for i in range(0, len(inputs)):
        weights[i] = weights[i] + learning_rate * (target - output) * inputs[i]
    return weights


# In[5]:


def train_perceptron(train_x, train_y, weights_init, iterations, learning_rate): #trains single perceptron weights
    weights = weights_init
    for i in range(0, iterations): # iterate through all training examples n times
        for j in range(0, len(train_x)):
            output = perceptron(train_x[j], weights)
            if output != train_y[j]: #only update weights if example is misclassified
                weights = update_weights(weights, train_x[j], train_y[j], output, learning_rate)
    training_results = []
    for j in train_x:
        training_results.append(perceptron(j, weights))
    return weights, training_results


# In[6]:


train_x = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
weights_init = [0, 0, 0] #initialize weights to 0
iterations = 10
learning_rate = .2


# In[8]:


boolean = {"false_" :      {"target": [-1, -1, -1, -1]}, 
           "and_" :        {"target": [-1, -1, -1, 1]}, 
           "a_and_not_b" : {"target": [-1, 1, -1, -1]},
            "a_":          {"target": [-1, 1, -1, 1]},
           "not_a_and_b" : {"target": [-1, -1, 1, -1]},
           "b_" :          {"target": [-1, -1, 1, 1]},
           "x_or" :        {"target": [-1, 1, 1, -1]},
           "or_" :         {"target": [-1, 1, 1, 1]},
           "nor_" :        {"target": [1, -1, -1, -1]},
           "x_nor" :       {"target": [1, -1, -1, 1]},
           "not_b" :       {"target": [1, 1, -1, -1]},
           "a_or_not_b" :  {"target": [1, 1, -1, 1]},
           "not_a" :       {"target": [1, -1, 1, -1]},
           "not_a_or_b":   {"target": [1, -1, 1, 1]},
           "nand" :        {"target": [1, 1, 1, -1]},
           "true_" :       {"target": [1, 1, 1, 1]}}


# In[9]:


results = {}
for i in boolean:
        boolean[i]["final_weights"], results[i] = train_perceptron(train_x, boolean[i]["target"], weights_init, iterations, learning_rate)
        if results[i] == boolean[i]["target"]:
            boolean[i]["Status"] = "Success"
        else:
            boolean[i]["Status"] = "Failure"


# In[10]:


df = pd.DataFrame(boolean).transpose()
df #shows that a single perceptron can represent any boolean valued function except for: "XOR" and "XNOR"

