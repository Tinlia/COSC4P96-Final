# The Random Forest Classifier for getting the fitness of the GA's chromosomes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from getdata import get

# Method for train/test using features selected by ga.py
def evaluate(c: tuple):
    # Get dataset

    # Select features based on chromosome from GA

    # Split into train:test
    
    # Train RFC

    # Predict and calculate accuracy
    pass

# Calculate the avg accuracy of a chromosome
def average_accuracy(c: tuple, runs=3):
    # Run eval thrice
    pass

