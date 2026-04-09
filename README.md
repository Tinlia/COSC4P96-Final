# COSC4P96-Final
This project uses Genetic Algorithms to optimize feature selection on the Pima Indian Diabetes Dataset, combined with either a Random Forest or KNN classifier for evaluating the fitness of feature subsets. Built for the final project of COSC 4P96.

## Dependencies
Running this project requires `matplotlib` for plotting fitness data and `scikit-learn` for the RFC/KNN implementation.

## Quick Start
1. Clone the repository into your editor
```bash
git clone https://github.com/Tinlia/COSC4P96-Final
cd COSC4P96-Final
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Dataset
Currently, the dataset being used in training is the Pima Indian Diabetes Dataset ([PIDD](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)). A lightweight dataset containing 8 features and 768 data points.

# File Overview
In total, there are two runnable files and three helper files.

## ga.py
This is the core file for optimizing feature selection in a database. `ga.py` contains the Genetic Algorithm for optimizing feature subsets and makes use of [getdata.py](#getdatapy) for pre-loading the database and [rfc.py](#rfcpy) for gathering the fitnesses of each chromosome.

## bruteforce.py
For smaller search spaces, `bruteforce.py` can be used to test every possible combination of feature subsets, with the exception of a "no feature" subset to avoid breaking the RFC. Like [ga.py](#gapy), it makes use of [getdata.py](#getdatapy) for pre-loading the database and [rfc.py](#rfcpy) for gathering the fitnesses of each chromosome.

## rfc.py
A helper file containing the implementation of a basic Random Forest Classifier. The average accuracy across three seeded runs produces the fitness for a chromosome.

```py
# Method for train/test using features selected by ga.py
def evaluate(c: tuple, seed: int) -> float:
    # RFC Logic

# Calculate the avg accuracy of a chromosome
def avg_accuracy_rfc(c: tuple, runs=3) -> float:
    acc = 0
    for i in range(runs): acc += evaluate(c, i)
    return float(acc / runs)
```

# knn.py
A helper file containing the implementation of a basic KNN classifier. Unlike [rfc.py](#rfcpy), the accuracy for a chromosome only needs to be evaluated once, since the current setup runs deterministically. This can be made stochastic by upping the number of `runs` in `avg_accuracy_knn()` and setting `random_state=1` to the `run` number.
```py
# Method for train/test using features selected by ga.py
def evaluate_knn(c: tuple, k=5: int) -> float:
    # ...
    x_train, ... = train_test_split(..., random_state=1) # random_state could be set to the run num to add stochasticity
    #...

# Calculate the avg accuracy of a chromosome
def avg_accuracy_knn(c: tuple, runs=1, k=5) -> float:
    acc = 0
    for r in range(runs): acc += evaluate_knn(c, r, k)
    return float(acc / runs)
```

## getdata.py
A helper file for fetching and loading the dataset into memory. 

```py
dataset = []

# Fetch and save the dataset to memory
def load():
    # ...
    
# Return the dataset
def get() -> list[list]:
    return dataset
```


# GA Structure
The Genetic Algorithm takes in the number of features in the dataset and produces chromosomes in the format `[0, 1, 1, 1, 0, 1, 0, 1]` with `0` representing a feature to omit and `1` representing a feature to keep.

## Fitness
Chromosome fitness is determined by the average accuracy of the model trained on the feature subset across 3 runs.

## Crossover
The GA uses single-point, two-parent crossover. For each pair of chromosomes, there is a `CROSSOVER_RATE` chance of crossover occuring. The resulting two children are added to the new population

## Mutation
All non-elite, non-diverse chromosomes have a `MUTATION_RATE` chance of mutating. If the mutation chance hits, each cell then has a `MUTATION_RATE` chance of flipping bits. The resulting chromosome is added to the new population.

## Elitism and Diversity
By default, 10% of the population is Elite and 10% is Diverse.

## Caching
The nature of this project demands high computational expense. As such, a dictionary which holds chromosome:fitness(key:value) pairs is used to prevent the redundant retraining of feature subsets that have already been evaluated. This practice is explained further [here](#determinism-vs-stochasticity).
```py
def fitness(c: tuple):
    # ...
    if cache.get(c) is not None: return cache[c]
    # ...
    fit = avg_accuracy(c, runs=3)
    cache[c] = fit
    return fit
```

# RFC Training
A simple Random Forest Classifier is used to determine the fitness of feature subsets. By default, the RFC has a max depth of 9 with 200 estimators. Due to the skew between positive and negative diagnoses within the [PIDD](#dataset)(~66% positive), we stratify the dataset when creating a test:train split.

## Determinism vs Stochasticity
Building on the idea of [caching](#caching), to keep computation expense low, we cache the fitnesses of feature subsets that have already been evaluated. This, of course, runs into the problem of potentially locking in a random solution and preventing a better one from appearing. To balance this out, we seed the train:test split to create deterministic data sampling. 
```py
x_train, x_test, y_train, y_test = train_test_split(..., random_state=1)
```
On the contrary, for the RFC, to keep an element of stochasticity, all training and testing is done on seeds 1-`k` where `k` is the number of times each subset is evaluated. The average of the `k` accuracies from training is then returned as the fitness. 
```py
rfc = RandomForestClassifier(..., random_state=seed) # Seed = run num
```
However, since the KNN classifier is distance-based and breaks any ties using the sum of distances for each class considered, ties *very* rarely occur. If they do, scikit-learn defaults to choosing the lower label, which in this case is `0: Non-diabetic`. Because of this, the knn is entirely deterministic. As explained in [knn.py](#knnpy), an element of stochasticity can be added by setting `random_state` to the run number.
