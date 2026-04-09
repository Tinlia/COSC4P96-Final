# Create every possible combination of features and test them with the classifier to get their fitnesses. 
from rfc import avg_accuracy_rfc
from getdata import load
from itertools import product
load()

def brute_force_search():
    best_score = 0
    best_mask = None
    results = []

    # Generate all 2^8 combinations
    for combo in product([0, 1], repeat=8):
        
        # Skip empty feature set
        if sum(combo) == 0:
            continue
        
        score = avg_accuracy_rfc(combo)
        results.append((combo, score))

        if score > best_score:
            best_score = score
            best_mask = combo

        print(f"Tested {combo} → Accuracy: {score:.4f}")

    print("\nBest Result:")
    print(f"Mask: {best_mask}")
    print(f"Accuracy: {best_score:.4f}")

    results.sort(key=lambda x: x[1], reverse=True) # Sort results by accuracy in descending order

    with open("outputs/brute_force_results.txt", "w") as f:
        for combo, score in results:
            f.write(f"{combo}, {score:.4f}\n")

    return best_mask, best_score, results

brute_force_search()