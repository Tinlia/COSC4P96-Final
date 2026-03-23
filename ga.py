# The Genetic Algorithm for creating chromosomes for feature selection
import random
from getdata import load
from classifier import avg_accuracy

# Load Dataset
load()
print("Dataset loaded successfully")
# Hyperparameters
CHROMOSOME_LENGTH = 8
POPULATION_SIZE = 30 
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
ELITISM_RATE = 0.1
DIVERSITY_RATE = 0.1
GENERATIONS = 100000 
TOURNAMENT_SIZE = 4
MAX_FITNESS = avg_accuracy((1, 1, 1, 1, 1, 1, 1, 1), runs=3) # The avg fitness of the chromosome with all features
COUNT = 0

# Storage
cache = {} # Caches the average fitness of each chromsome to prevent excessive training (chromosome -> fitness)

# Fitness function
def fitness(c: tuple):
    # Check cache for fitness and return it if it exists
    if cache.get(c) is not None:
        return cache[c]
    
    # Punish for 8&0 features
    if sum(c) == 8 or sum(c) == 0: # If all or none of the features are selected, punish heavily
        return 0
    
    # Else, run it thrice and take the average
    fit = avg_accuracy(c, runs=3)

    # Cache fitness and return
    cache[c] = fit
    return fit

# Mutation method
def mutation(chromosome):
    # Create a copy of the chromosome
    mutated = list(chromosome)

    # Mutation core
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < MUTATION_RATE: # Odds of cell mutation = Odds of chromosome mutation
            mutated[i] = 1 - mutated[i] # Flips between 0 and 1

    return tuple(mutated) # Return the mutated chromosome

# Crossover Method
def crossover(p1, p2):
    split = random.randint(1, CHROMOSOME_LENGTH - 1) # Point to split on
    c1 = p1[:split] + p2[split:] # Child 1 
    c2 = p2[:split] + p1[split:] # Child 2
    return [c1, c2] # Return the two child chromosomes

def tournament_selection(pops):
    # Intake a list of k chromosomes with their fitnesses attached
    pops.sort(key=lambda x: x[1], reverse=True) # Sort by fitness
    return pops[0][0] # Return the chromosome with the highest fitness

def gen_chromosome() -> tuple:
    c = [0] * CHROMOSOME_LENGTH
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < 0.5:
            c[i] = 1
    return tuple(c)

# Gen Random Pop
pop = []
for _ in range(POPULATION_SIZE):
    pop.append(tuple((gen_chromosome(), 0))) # (chromosome, fitness)

# Run generations
for g in range(GENERATIONS):
    # Fetch fitnesses
    for i in range(POPULATION_SIZE):
        fit = fitness(pop[i][0])
        pop[i] = (pop[i][0], fit) # Update fitness in population
    
    # Sort fitnesses desc.
    pop.sort(key=lambda x: x[1], reverse=True) 

    # Break condition for max fitness
    if pop[0][1] >= MAX_FITNESS:
        print(f"Relative accuracy achieved at generation {g} with chromosome {str(pop[0][0])}")
        break
    new_pop = []

    # Elitism
    enum = int(ELITISM_RATE * POPULATION_SIZE)
    for i in range(enum):
        new_pop.append(pop[i])

    # Crossover
    while len(new_pop) < POPULATION_SIZE - int(DIVERSITY_RATE * POPULATION_SIZE):
        p1 = tournament_selection(random.sample(pop, TOURNAMENT_SIZE)) # Tournament selection on k random chromosomes
        p2 = tournament_selection(random.sample(pop, TOURNAMENT_SIZE)) # Select 5 random chromosomes and pick the best one as parent 2
        if random.random() < CROSSOVER_RATE:
            c1, c2 = crossover(p1, p2)
        else:
            c1, c2 = p1, p2
        new_pop.append((c1, 0))
        new_pop.append((c2, 0))
    
    # Diversity
    while len(new_pop) < POPULATION_SIZE:
        new_pop.append((gen_chromosome(), 0))

    while len(new_pop) > POPULATION_SIZE:
        print("Pop Trimmed")
        new_pop.pop() # Remove excess chromosomes if we went over population size

    # Mutate
    for i in range(enum, len(new_pop)-int(DIVERSITY_RATE * POPULATION_SIZE)): # Don't mutate elites or diversity chromosomes
        if random.random() < MUTATION_RATE:
            new_pop[i] = (mutation(new_pop[i][0]), 0.85)

    # Update pop
    pop = new_pop