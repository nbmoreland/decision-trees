# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a decision tree implementation for machine learning classification tasks. The main file `decision_tree.py` implements the DTL (Decision Tree Learning) algorithm with support for both optimized and randomized attribute selection.

## Core Architecture

### Main Components

1. **DecisionTree Class** (`decision_tree.py:8-15`): Node structure for the decision tree
   - Stores attribute index, threshold value, left/right children, and information gain

2. **DTL Algorithm** (`decision_tree.py:139-152`): Recursive decision tree construction
   - Implements pruning based on minimum sample threshold
   - Supports two modes: "optimized" (exhaustive search) and "randomized" (random attribute selection)

3. **Information Gain Calculation** (`decision_tree.py:41-72`): Entropy-based splitting criterion
   - Uses binary splits based on numerical thresholds

### Key Functions

- `decision_tree()` (`decision_tree.py:164-213`): Main entry point that handles training and testing
- `choose_attribute()` (`decision_tree.py:75-105`): Selects best attribute and threshold for splitting
- `probability()` (`decision_tree.py:108-114`): Traverses tree to get class probability distribution

## Running the Code

The main function expects:
```python
decision_tree(training_file, test_file, option, pruning_thr)
```

Parameters:
- `training_file`: Path to training data file (space-separated values, last column is class label)
- `test_file`: Path to test data file (same format)
- `option`: Either "optimized", "randomized", or integer (1 or 3) for ensemble methods
- `pruning_thr`: Minimum number of samples required to create a split

## Data Format

Input files should contain:
- Space-separated numerical values
- Last column contains class labels
- Each row represents one example

## Testing

Currently no automated tests are present. To test:
1. Prepare training and test data files in the required format
2. Call the `decision_tree()` function with appropriate parameters
3. Output includes tree structure and classification accuracy