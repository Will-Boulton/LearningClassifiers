from __future__ import division
from sample import Sample
from Classifier import Classifier
import sys


class DecisionTreeClassifier(Classifier):

    def __init__(self, max_depth, min_group_size):
        """
        Args:
            max_depth: Maximum Depth of Decision Tree
            min_group_size: Smallest number of samples before stopping splitting a node
        """
        Classifier.__init__(self)
        self.min_group_size = min_group_size
        self.max_depth = max_depth
        self.root = None
        self.nodes = []

    def split_for_feature(self, feature_index, feature_value, group_to_split):
        """ splits a group of feature vectors at an index and given value of the feature vector at that index
        Args:
            feature_index: index of feature being considered within each feature vector
            feature_value:  value of feature to split around
            group_to_split: feature vector to split into groups
        """
        left = [point for point in group_to_split if point.features[feature_index] < feature_value]
        right = [point for point in group_to_split if point not in left]
        return left, right

    def gini(self, groups, labels):
        """ Calcualtes the gini impurity of a split of a dataset
        Args:
            groups: list of groups of samples
            labels: labels associated with the groups
        """
        n = sum([len(group) for group in groups])
        gini_impurity = 0.0
        for group in groups:
            group_size = len(group)
            if group_size != 0:
                score = 0.0
                for label in labels:
                    frac_positive = [data_point.class_ for data_point in group].count(
                        label) / group_size  # P(C) = positive / (positive + negative)
                    score += frac_positive ** 2
                gini_impurity += (1 - score) * (group_size / n)
        return gini_impurity

    def best_split(self, group_to_split):
        """ Finds the best split of a group by it's gini index
        Args:
            group_to_split: group to split into two by gini index
        """
        labels = list(set(point.class_ for point in group_to_split))
        best_gini, best_value, best_score, best_groups = sys.maxint, sys.maxint, sys.maxint, None
        for feature_index in range(
                len(group_to_split[0].features)):  # try splitting by each feature and each value it takes in the group
            for point in group_to_split:
                # split based on a feature
                groups = self.split_for_feature(feature_index, point.features[feature_index], group_to_split)
                gini = self.gini(groups, labels)  # work out the gini index of this split
                if gini < best_score:  # this split is the current best tried
                    best_gini, best_value, best_score, best_groups = feature_index, point.features[
                        feature_index], gini, groups
        return (best_groups, best_gini, best_value)  # make a node that contains this split's information

    def train(self, retrain=False):
        """ Trains the DecisionTreeClassifier by generating a DecisionTree
        Args:
            retrain: whether or not the model is already trained and therefore needs to restart from scratch
        """
        if self.root and not retrain:
            raise Exception("DECISION TREE::ERROR::ALREADY TRAINED CLASSIFIER AND RETRAIN PARAMETER IS FALSE")
        print "Generating Decision Tree..."
        groups, index, value = self.best_split(self.points)
        self.root = TreeNode(groups, index, value)
        self.nodes.append(self.root)
        print "..."
        tree = self.make_tree(self.root, 1, self.max_depth, self.min_group_size)
        print "Done"
        return tree

    def make_tree(self, node, depth, max_depth, min_size):
        """ recursively makes the sub tree from node, this tree is used to classify samples
        Args:
            node: node to make subtree from
            depth:  current depth of node in tree
            max_depth:  maximum depth of node in tree
            min_size:   minimium size of a split of a group of feature vectors
        """
        left, right = node.groups

        sz_left, sz_right = len(left), len(right)

        if sz_left == 0 or sz_right == 0: #if the split made one empty group then this node should be a leaf
            node.make_leaf()
            node.left_child = node
            node.right_child = node
            node.is_leaf = True
            return

        if depth >= max_depth:             #if the tree is too deep then make the node a leaf
            node.make_leaf()
            node.left_child = node
            node.right_child = node
            node.is_leaf = True
            return

        #process left subtree
        if sz_left <= min_size:             #if the group is too small make the node into a leaf
            left_c = TreeNode((left, []), depth=depth)
            node.left_child = (left_c)
            self.nodes.append(left_c)
            left_c.make_leaf()

            right_c = TreeNode(([], right), depth=depth)
            node.set_right(right_c)
            self.nodes.append(right_c)
            right_c.make_leaf()
        else:                               #otherwise  a left child node that isnt't a leaf and process it
            grp, ind, val = self.best_split(left)
            left_c = TreeNode(grp, ind, val, depth=depth)
            node.left_child = (left_c)
            self.nodes.append(left_c)
            self.make_tree(left_c, depth + 1, max_depth, min_size)

        #do the same on the right subtree
        if sz_right <= min_size:
            right_c = TreeNode(([], right), depth=depth)
            node.set_right(right_c)
            self.nodes.append(right_c)
            right_c.make_leaf()

            left_c = TreeNode((left, []), depth=depth)
            node.left_child = (left_c)
            self.nodes.append(left_c)
            left_c.make_leaf()
        else:
            grp, ind, val = self.best_split(right)
            right_c = TreeNode(grp, ind, val, depth=depth)
            node.right_child = right_c
            self.nodes.append(right_c)
            self.make_tree(right_c, depth + 1, max_depth, min_size)

    def predict(self, in_point, node):
        """ Predicts the class of a point recursively by searching for a leaf node that matches the value of the point
        Args:
            in_point: sample to classify
            node:   current subtree root to consider recursively
        """
        if node.is_leaf:
            return node.value
        if in_point.features[node.feature_index] < node.value:

            return self.predict(in_point, node.left_child)
        else:
            return self.predict(in_point, node.right_child)

    def classify(self, inpoint):
        """ Classifies a sample by calling predict on the root of the DecisionTree
        Args:
            inpoint: Sample to classify
        """
        if not self.root:
            raise Exception("DECISION TREE::ERROR:CLASSIFIER UNTRAINED")
        class_ = self.predict(inpoint, self.root)
        inpoint.predicted_class = class_
        return class_


class TreeNode:
    id_ = 0

    def __init__(self, groups, feature_index=False, value=False, leaf=False, depth=0):
        """
        Args:
            groups: the split dataset used for making child nodes
            feature_index:  index of feature within the feature vector used to split the data
            value:  value of feature at index as a bound to decide whether to traverse left or right subtree in prediction
            leaf:   whether this node is a leaf
            depth:  how deep down the whole tree is this node
        """
        self.feature_index = feature_index
        self.value = value
        self.groups = groups
        self.left_child = None
        self.right_child = None
        self.is_leaf = False
        self.depth = depth
        if leaf:
            self.make_leaf()
        self.id = TreeNode.id_
        TreeNode.id_ += 1

    def set_left(self, left):
        """ sets left child to a node
        Args:
            left:   node to be set as left child
        """
        self.left_child = left

    def set_right(self, right):
        """sets right child to a node
        Args:
            right: node to be set as right child
        """
        self.right_child = right

    def make_leaf(self):
        """Makes this node into a leaf by making children refer back to itself
        """
        self.is_leaf = True
        self.left_child, self.right_child = self, self
        self.value = self.get_leaf(self.groups[0] + self.groups[1])

    def __repr__(self):
        string = ""
        if self.is_leaf:
            string = 'class' + str(self.value)
        else:
            string = 'F:' + str(self.feature_index + 1) + ' < ' + str(self.value)
        return string

    def __str__(self):
        l = str(self.groups[0]) + str(self.is_leaf) + str(self.depth)  # if self.left_child is not None else "x"
        r = str(self.groups[1]) + str(self.is_leaf) + str(self.depth)  # if self.right_child is not None else "x"
        y = ''
        if not self.left_child or not self.right_child: y = " THIS ONE BROKED"
        return str(l) + "," + str(r) + y

    def get_leaf(self, group):
        """ returns value of the leaf as the most common class in the group
        Args:
            group: group to search for most common class
        """
        classes = [point.class_ for point in group]
        return max(set(classes), key=classes.count)

