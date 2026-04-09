from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from getdata import get

# KNN classifier
def evaluate_knn(c: tuple, r: int, k: int) -> float:
    dataset = get()
    x = [row[:-1] for row in dataset] # Feature list
    y = [row[-1] for row in dataset]  # Label List

    # Select feature indices from chromosome
    selected_features = [i for i, bit in enumerate(c) if bit == 1]

    # Keep only selected features
    x_sel = [[row[i] for i in selected_features] for row in x]

    # Fixed split for fair comparison (train:test - 80:20)
    x_train, x_test, y_train, y_test = train_test_split(x_sel, y, test_size=0.2, stratify=y, random_state=1) # Due to the skewed labels, stratify to balance the train:test

    # Because knn is distance-based, scale the features to ensure fair distance calculations
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(x_train, y_train)

    # Predict
    y_pred = knn.predict(x_test)

    return float(accuracy_score(y_test, y_pred))

# Run the knn a few times to get the average acc and let that be the fitness
def avg_accuracy_knn(c: tuple, runs=1, k=5) -> float:
    # A note for why this exists, what with the knn being entirely deterministic in its current state:
    #  this is kept here so that if you want to add stochasticity to the knn (i.e., setting `random_state` in the t:t split to the run num)
    #  you can still test the avg accuracy without having to readd the avg accuracy function and refactor everything in ga.py.

    # Also, just changing "_rfc" to "_knn" for the CLASSIFIER hyparam in ga.py is so much easier
    acc = 0
    for r in range(runs):
        acc += evaluate_knn(c, r, k)
    return float(acc / runs)