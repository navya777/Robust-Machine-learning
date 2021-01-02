# Robust-Machine-learning
Updating hinge loss using data augmentation to implement robustness




Train a hinge loss that will accurately classify large distortions (corruptions) in the 
MNIST dataset posted below this exercise link. 

Submit your robust hinge loss training program called "robust_hinge.py". The inputs 
to your program are the MNIST train dataset and labels for class 0 vs 1 posted below 
this exercise link. Your hinge loss will output the w vector and w0 in one file 
called "w_vector". The first line contains the vector with entries separated by space 
and the next line contains the w0 value.

We will grade your solution by training your submitted program on the MNIST dataset. 
We will then use the outputted w_vector to evaluate the accuracy of the corrupted 
data posted on the website below this exercise.

To get full points your robust classifier must 

1. achieve low error on the clean test dataset (at most 1%)

2. achieve low error (below 1%) on at least three of the MNIST 
   corruptions: fog, brightness, stripe, scale, and translate. It is 
   possible that some corruptions are complementary, in the sense that one 
   set of model parameters may give low error on fog but high on translate 
   and another set would be vice-versa.
