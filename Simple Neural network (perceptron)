
import numpy as np
 

def sigmoid(x): 
    return  1/(1+np.exp(-x)) 

 

def sigmoid_derivation(x):
    return x*(x-1)
input=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]]) 

 

a_output=np.array([[0,1,1,0]]).T 

 

np.random.seed(1)
weights=2*np.random.random((3,1))-1
print('Initial weights are:')
print(weights)
for i in range(10000): 
    input_layer=input 
    o_output=sigmoid(np.dot(input_layer,weights)) 
    loss=o_output-a_output 
adjustment=loss*sigmoid_derivation(o_output) 
weights=weights+np.dot(input_layer.T,adjustment)
print('new weights are:')

print(weights) 

 

print('obtained output values are') 

print(o_output) 
