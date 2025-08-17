# Nicholas Moreland
# 1001866051

import numpy as np
import random

# Decision Tree Node
class DecisionTree:
    def __init__(self, attribute=-1, threshold=-1, left=None, right=None, data=-1, gain=0):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.data = data
        self.gain = gain
    
# Gets class labels
def get_classes(examples):
    classes = []
    counter = 0
    for i in range(len(examples)):
        if examples[i][-1] not in classes:
            classes.append(examples[i][-1])
    count = dict.fromkeys(classes, 0)
    for i in sorted(count.keys()):
        count[i] = counter
        counter += 1
    return count

# Return probability distribution of classes in examples
def class_distribution(examples):
    distribution_array = np.zeros(len(distribution))
    for i in range(len(examples)):
        distribution_array[distribution[examples[i][-1]]] += 1
    for i in range(len(distribution_array)):
        if len(examples) > 0:
            distribution_array[i] /= len(examples)
    return distribution_array

# Calculate information gain of the data
def information_gain(examples, attribute, threshold):
    examples_left = []
    examples_right = []
    H_E = H_E1 = H_E2 = 0

    for example in examples:
        if example[attribute] < threshold:
            examples_left.append(example)
        else:
            examples_right.append(example)

    dist = class_distribution(examples)
    dist_left = class_distribution(examples_left)
    dist_right = class_distribution(examples_right)

    for num in dist:
        if num > 0:
            H_E -= (num * np.log2(num))
    
    for num in dist_left:
        if num > 0:
            H_E1 -= (num * np.log2(num))

    for num in dist_right:
        if num > 0:
            H_E2 -= (num * np.log2(num))

    K = len(examples)
    K1 = len(examples_left)
    K2 = len(examples_right)
    result = H_E - ((K1 / K) * H_E1) - ((K2 / K) * H_E2)
    return result

# Choose the attribute with the highest information gain
def choose_attribute(option, attributes, examples):
    if option == "optimized":
        max_gain = best_attribute = best_threshold = -1

        for attribute in attributes:
            attribute_values = [x[attribute] for x in examples]
            L = min(attribute_values)
            M = max(attribute_values)

            for k in range(1, 51):
                threshold = L + (k * (M - L) / 51)
                gain = information_gain(examples, attribute, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = attribute
                    best_threshold = threshold
        return (best_attribute, best_threshold, max_gain)
    elif option == "randomized":
        max_gain = best_threshold = -1
        attribute = random.choice(attributes)
        attribute_values = [x[attribute] for x in examples]
        L = min(attribute_values)
        M = max(attribute_values)

        for k in range(1, 51):
            threshold = L + (k * (M - L) / 51)
            gain = information_gain(examples, attribute, threshold)
            if gain > max_gain:
                max_gain = gain
                best_threshold = threshold
        return (attribute, best_threshold, max_gain)

# Probability class label
def probability(tree, test_data):
    if tree.left == None and tree.right == None:
        return tree.data
    if test_data[tree.attribute] < tree.threshold:
        return probability(tree.left, test_data)
    else:
        return probability(tree.right, test_data)
    
# Print the tree in a depth-first children left to right
def print_breath_first(parent, tree_id, node_n):
    if not parent:
        return

    queue = []
    queue.append(parent)

    while queue:
        temp_node = queue.pop(0)
        print("tree=%2d, node=%3d,feature=%2d, thr=%6.2f, gain=%f" % (tree_id + 1, node_n,temp_node.attribute, temp_node.threshold, temp_node.gain))

        node_n += 1
        if temp_node.left:
            queue.append(temp_node.left)
        if temp_node.right:
            queue.append(temp_node.right)

# The top level function for the DTL algorithm
def DTL_TopLevel(examples, attributes, pruning_thr, option):
    default = class_distribution(examples)[1]
    return DTL(examples, attributes, default, pruning_thr, option)

# Returns a new decision tree based on the examples and attributes given
def DTL(examples, attributes, default, pruning_thr, option):
    if len(examples) < pruning_thr:
        return DecisionTree(data=default)
    elif 1 in class_distribution(examples):
        return DecisionTree(data=class_distribution(examples))
    else:
        (best_a, best_t, gain) = choose_attribute(option, attributes, examples)
        tree = DecisionTree(best_a, best_t, gain=gain)
        examples_left = [x for x in examples if x[best_a] < best_t]
        examples_right = [x for x in examples if x[best_a] >= best_t]
        tree.left = DTL(examples_left, attributes, class_distribution(examples), pruning_thr, option)
        tree.right = DTL(examples_right, attributes, class_distribution(examples), pruning_thr, option)
        return tree

# Load data from a file path 
def load_data(path):
    array = []
    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            array.append(split)
    return np.array(array).astype(np.float64)


def decision_tree(training_file, test_file, option, pruning_thr):
    global distribution

    # Load data
    training_data = load_data(training_file)
    test_data = load_data(test_file)
    attributes = range(len(training_data[0][:-1]))

    # Training Phase
    distribution = get_classes(training_data)
    trees = []

    if option == "optimized" or option == "randomized":
        trees.append(DTL_TopLevel(training_data, attributes, pruning_thr, option))
    else:
        if option == 1:
            option = "randomized"
            for i in range(1):
                trees.append(DTL_TopLevel(training_data, attributes, pruning_thr, option))
        elif option == 3:
            option = "randomized"
            for i in range(3):
                trees.append(DTL_TopLevel(training_data, attributes, pruning_thr, option))

    # Print the tree in a depth-first children left to right
    for i in range(len(trees)):
        print_breath_first(trees[i], i, node_n=1)

    # Testing Phase
    correct_count = 0

    for n in range(len(test_data)):
        accuracy = 0
        distance = []

        for i in range(len(trees)):
            distance.append(probability(trees[i], test_data[n]))

        index = np.argmax(distance)
        index_class = index

        if len(distance[0]) < index:
            index_class = index % len(distance[0])
        if index_class == test_data[n][-1]:
            accuracy = 1
        
        correct_count += accuracy
        print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f" % (n+1, index_class, int(test_data[n][-1]), accuracy))
    
    print("classification accuracy= %6.4f" % (correct_count / len(test_data)))