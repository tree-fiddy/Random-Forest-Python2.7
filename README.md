# Random Forests
Author:  David Yun

## Introduction
This exploration into random forests uses the [UCI Credit Approval Dataset](https://archive.ics.uci.edu/ml/datasets/Credit+Approval).
Rows with missing attributes were removed.  

### Readings and Resources
To understand random forests, particularly entropy and information gain,
the following presentation [from CMU was used](http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf).  

Another great resource is Chapter 15 in [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)-
a fantastic textbook, which happens to be free.  

To run the program, simply type random_forest.py after writing all the necessary classes and methods, and specify tree size
to calculate the accuracy of the model.  

As I was typing up the classes and methods, the following hints were essential:  
-  Which attributes to use when building a tree?
-  How to determine the split point for an attribute?  
-  When do you stop splitting leaf nodes?
-  How many trees should the forest contain?

## Files
### util.py
-  Consists of:
    1. Entropy function
    2. Partition Classes function
    3. Information Gain
    
### Decision_tree.py
-  Consists of:
    1.  learn function
    2.  Split function
    3.  to terminal function
        -  If no more split, assign as terminal node
    4.  check if nodes are identical
    5.  Create splitting of nodes
    6.  Classify
    7.  Predict
    
### Random_Forest.py
-  Consists of:
    1.  Various functions for bootstrapping
    2.  Calculate Out of Bag (OOB) errors
    3.  Calculate Accuracy via voting
    
    
    