# Decision Tree Classifier

A Python implementation of the Decision Tree Learning (DTL) algorithm for classification tasks, featuring both optimized and randomized attribute selection strategies.

## Overview

This project implements a decision tree classifier from scratch using information gain (entropy) as the splitting criterion. The implementation supports:
- Binary splits on continuous attributes
- Pre-pruning based on minimum sample size
- Both exhaustive and randomized attribute selection
- Ensemble methods with multiple trees

## Features

- **Optimized Mode**: Exhaustively searches all attributes and thresholds to find the best split
- **Randomized Mode**: Randomly selects attributes for splitting (useful for ensemble methods)
- **Ensemble Support**: Can train multiple trees for improved accuracy
- **Flexible Pruning**: Configurable minimum sample threshold to prevent overfitting
- **Detailed Output**: Provides tree structure visualization and per-instance accuracy metrics

## Installation

### Requirements
- Python 3.x
- NumPy

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/decision-trees.git
cd decision-trees

# Install dependencies
pip install numpy
```

## Usage

### Basic Usage

```python
from decision_tree import decision_tree

# Run decision tree classification
decision_tree(
    training_file="path/to/training_data.txt",
    test_file="path/to/test_data.txt", 
    option="optimized",  # or "randomized", 1, 3
    pruning_thr=50       # minimum samples for splitting
)
```

### Parameters

- **training_file** (str): Path to the training data file
- **test_file** (str): Path to the test data file
- **option** (str or int): Training mode
  - `"optimized"`: Finds the best attribute and threshold at each split
  - `"randomized"`: Randomly selects attribute, then finds best threshold
  - `1`: Single randomized tree
  - `3`: Ensemble of 3 randomized trees
- **pruning_thr** (int): Minimum number of samples required to create a split (pre-pruning parameter)

### Data Format

Input data files should be formatted as:
- Space-separated numerical values
- Each row represents one example
- Last column contains the class label (integer)
- All features must be continuous/numerical

Example data format:
```
5.1 3.5 1.4 0.2 0
4.9 3.0 1.4 0.2 0
6.2 3.4 5.4 2.3 2
5.9 3.0 5.1 1.8 2
```

## Algorithm Details

### Information Gain

The algorithm uses entropy-based information gain to determine the best splits:

```
H(S) = -Σ p(c) * log₂(p(c))
IG(S, A) = H(S) - Σ |Sv|/|S| * H(Sv)
```

Where:
- H(S) is the entropy of set S
- p(c) is the proportion of samples belonging to class c
- IG(S, A) is the information gain from splitting on attribute A

### Splitting Strategy

For each attribute:
1. Determines the range [L, M] of attribute values in the current dataset
2. Tests 50 evenly-spaced thresholds between L and M
3. Selects the threshold that maximizes information gain

### Tree Construction

The DTL algorithm recursively:
1. Checks stopping conditions (pruning threshold or pure node)
2. Selects the best attribute and threshold (based on mode)
3. Splits the data into left (< threshold) and right (≥ threshold) subsets
4. Recursively builds subtrees

## Output Format

### Tree Structure Output
```
tree= 1, node=  1, feature= 2, thr=  2.45, gain=0.811278
tree= 1, node=  2, feature= 3, thr=  1.75, gain=0.444392
...
```

### Classification Results
```
ID=    1, predicted=  0, true=  0, accuracy=1.00
ID=    2, predicted=  1, true=  1, accuracy=1.00
...
classification accuracy= 0.9533
```

## Code Structure

```
decision-trees/
├── decision_tree.py    # Main implementation
├── CLAUDE.md          # Development guide for Claude Code
└── README.md          # This file
```

### Key Classes and Functions

- **DecisionTree**: Node class for tree structure
- **decision_tree()**: Main entry point for training and testing
- **DTL()**: Recursive tree construction algorithm
- **choose_attribute()**: Attribute selection (optimized vs randomized)
- **information_gain()**: Calculates entropy-based information gain
- **probability()**: Tree traversal for prediction

## Example

```python
# Example with iris-like dataset
from decision_tree import decision_tree

# Train an optimized decision tree with pruning
decision_tree(
    training_file="iris_train.txt",
    test_file="iris_test.txt",
    option="optimized",
    pruning_thr=10
)

# Train an ensemble of 3 randomized trees
decision_tree(
    training_file="iris_train.txt",
    test_file="iris_test.txt",
    option=3,
    pruning_thr=10
)
```

## Performance Considerations

- **Time Complexity**: O(n²m) for optimized mode, O(nm) for randomized mode
  - n = number of samples
  - m = number of attributes
- **Space Complexity**: O(n) for tree storage
- **Pruning Impact**: Higher pruning thresholds reduce overfitting but may decrease training accuracy

## Limitations

- Only supports numerical/continuous attributes
- Binary splits only (no multi-way splits)
- No post-pruning implementation
- No built-in cross-validation
- Requires all data to fit in memory

## Future Enhancements

Potential improvements for contributors:
- [ ] Support for categorical attributes
- [ ] Post-pruning methods
- [ ] Additional splitting criteria (Gini index, gain ratio)
- [ ] Cross-validation functionality
- [ ] Feature importance calculation
- [ ] Tree visualization
- [ ] Parallel training for ensemble methods
- [ ] Support for missing values

## Author

Nicholas Moreland

## License

This project is available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## References

- Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106.
- Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.