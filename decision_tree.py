from util import entropy, information_gain, partition_classes, try_partition_classes
import numpy as np 
import ast
import csv

'''
Help from:
Overall Structure: http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html?page=3
https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py
https://gist.github.com/iamaziz/02491e36490eb05a30f8

https://codereview.stackexchange.com/questions/109089/id3-decision-tree-in-python
'''
class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        self.threshold = 0.1



    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        gain, index, value = self.try_split(X, y)
        self.tree = self.get_split(X, y, index, value)
        #print(self.tree)
        self.split(self.tree, 1)

    # Select the best split point for a dataset
    def try_split(self, X, y):
        b_index, b_value, b_gain = -1, -1, float('-inf')       
        
        X_arr = np.asarray(X, dtype=object)
        number_of_attr = X_arr.shape[1]
        idx = np.random.choice(range(number_of_attr), int(np.floor(0.4*number_of_attr)), replace=False)
        for index in idx:
            partition_values = np.unique(X_arr[:, index], return_counts=False)
            if not type(partition_values[0])==str:
                partition_values = np.percentile(partition_values, [10*i for i in range(1,10, 2)])
            
            gain = np.array([information_gain(y, try_partition_classes(X, y, index, value)) for value in partition_values])
            max_gain = max(gain)
            if max_gain < self.threshold:
                continue
            else:
                if max_gain > b_gain:
                    b_index, b_value, b_gain = index, partition_values[np.argmax(gain)], max_gain

        return (b_gain, b_index, b_value)

    def get_split(self, X, y, index, value):
        groups_xleft, groups_xright, groups_yleft, groups_yright = partition_classes(X, y, index, value)
        return {'index':index, 'value':value, 'groups_xleft':groups_xleft, 'groups_xright':groups_xright,\
                      'groups_yleft':groups_yleft, 'groups_yright':groups_yright}

    # Create a terminal node value
    def to_terminal(self, y):
        return max(set(y), key=y.count)

    def check_identical(seft, X):
        X_array = np.asarray(X, dtype=object)
        for m in range(X_array.shape[1]):
            if len(set(X_array[:, m])) > 1:
                return False
        return True


    # Create child splits for a node or make terminal
    def split(self, node, depth, max_depth=5, min_size=5):
        left_X, right_X, left_y, right_y = node['groups_xleft'], node['groups_xright'], node['groups_yleft'],  node['groups_yright']
        #del(node['groups_xleft'], node['groups_xright'], node['groups_yleft'],  node['groups_yright'])
        # check for a no split
        if(len(left_y)==0 or len(right_y)==0):
             node['left'] = node['right'] = self.to_terminal(left_y + right_y)
             return
        if(len(set(left_y + right_y))==1):
             node['left'] = node['right'] = self.to_terminal(left_y + right_y)
             return
        if(self.check_identical(left_X + right_X)):
             node['left'] = node['right'] = self.to_terminal(left_y + right_y)
             return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left_y), self.to_terminal(right_y)
            return
        # process left child
        if len(left_X) <= min_size:
            node['left'] = self.to_terminal(left_y)
        else:
            gain, index, value = self.try_split(left_X, left_y)
            if gain > self.threshold:
                node['left'] = self.get_split(left_X, left_y, index, value)
                self.split(node['left'], depth+1, max_depth, min_size)
            else:
                node['left'] = self.to_terminal(left_y)
        # process right child
        if len(right_X) <= min_size:
            node['right'] = self.to_terminal(right_y)
        else:
            gain, index, value = self.try_split(right_X, right_y)
            if gain > self.threshold:
                node['right'] = self.get_split(right_X, right_y, index, value)
                self.split(node['right'], depth+1, max_depth, min_size)
            else:
                node['right'] = self.to_terminal(right_y)
 


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        node = self.tree
        return self.predict(node, record)


    def predict(self, node, record):
        if isinstance(record[node['index']], str):   
            if record[node['index']] == node['value']:
                if isinstance(node['left'], dict):
                    return self.predict(node['left'], record)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return self.predict(node['right'], record)
                else:
                    return node['right']
        else:
            if record[node['index']] < node['value']:
                if isinstance(node['left'], dict):
                    return self.predict(node['left'], record)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return self.predict(node['right'], record)
                else:
                    return node['right']
