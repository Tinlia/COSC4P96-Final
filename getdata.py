# Dataset Loader
dataset = []

def load():
    global dataset
    with open("diabetes.csv", "r") as f:
        next(f) # Skip header
        for line in f:
            dataset.append(line.strip().split(","))

def get() -> list[list]:
    return dataset