# The Random Forest Classifier for getting the fitness of the GA's chromosomes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from getdata import get

# Method for train/test using features selected by ga.py
def evaluate(c: tuple, seed) -> float:
    # Get dataset and split into x and y
    dataset = get()
    x = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    # Select features based on chromosome from GA
    fts = [i for i in range(len(c)) if c[i] == 1]
    x_sel = [[row[i] for i in fts] for row in x]

    # Split into train:test, 80:20
    x_train, x_test, y_train, y_test = train_test_split(x_sel, y, test_size=0.2, stratify=y, random_state=1) # Due to unbalanced labels, stratify the dataset to balance it out
    
    # Train RFC
    rfc = RandomForestClassifier(n_estimators=200, max_depth=9, random_state=seed)
    rfc.fit(x_train, y_train)

    # Predict and calculate accuracy
    y_pred = rfc.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return float(accuracy)

# Calculate the avg accuracy of a chromosome
def avg_accuracy(c: tuple, runs=3) -> float:
    acc = 0

    # Run eval thrice
    for i in range(runs):
        acc += evaluate(c, i) # run num = seed for reproducibility
    return float(acc / runs)
