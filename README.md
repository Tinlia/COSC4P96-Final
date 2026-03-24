# COSC4P96-Final
This project uses Genetic Algorithms to optimize feature selection on the Pima Indian Diabetes Dataset, combined with a simple Random Forest Classifier for evaluating the fitness of feature subsets. Built for the final project of COSC 4P96

## Dependencies
Running this project requires `matplotlib` for plotting fitness data and `scikit-learn` for the RFC implementation.

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

# File Overview
In total, there are two runnable files and two helper files.

## ga.py
This is the core file for optimizing feature selection in a database. `ga.py` contains the Genetic Algorithm for optimizing feature subsets and makes use of [getdata.py](#getdatapy) for pre-loading the database and [classifier.py](#classifierpy) for gathering the fitnesses of each chromosome.

## bruteforce.py
For smaller search spaces, `bruteforce.py` can be used to test every possible combination of feature subsets, with the exception of a "no feature" subset to avoid breaking the RFC. Like [ga.py](#gapy), it makes use of [getdata.py](#getdatapy) for pre-loading the database and [classifier.py](#classifierpy) for gathering the fitnesses of each chromosome.

## classifier.py
A helper file containing the implementation of a basic Random Forest Classifier. The average accuracy across three seeded runs produces the fitness for a chromosome.

```py
# Method for train/test using features selected by ga.py
def evaluate(c: tuple, seed: int) -> float:
    # RFC Logic

# Calculate the avg accuracy of a chromosome
def avg_accuracy(c: tuple, runs=3) -> float:
    for i in range(runs): acc += evaluate(c, i)
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

