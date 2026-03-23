# The Genetic Algorithm for creating chromosomes for feature selection
import random

# Hyperparameters
CHROMOSOME_LENGTH = 8
POPULATION_SIZE = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
ELITISM_RATE = 0.1
GENERATIONS = 100 
TOURNAMENT_SIZE = 4
MAX_FITNESS = 1.0 # To be changed to the average accuracy of an 8-feature run of the model

# Storage
cache = {} # Caches the average fitness of each chromsome to prevent excessive training (chromosome -> fitness)

# Placeholder fitness function for fitness generation
def placeholder(c): 
    return random.random()

# Fitness function
def fitness(c: tuple):
    # Check cache for fitness and return it if it exists
    if cache.get(c) is not None:
        return cache[c]
    
    # Else, run it thrice and take the average
    fit = 0
    for _ in range(3):
        fit += placeholder(c) # Get the accuracy of the model with the features selected by the chromosome
    fit /= 3

    # Punish for more features, reward for less features
    if sum(c) == 8 or sum(c) == 0: # If all or none of the features are selected, punish heavily
        fit = 0

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

# Confirm Hyparams
if ELITISM_RATE + MUTATION_RATE + CROSSOVER_RATE != 1.0:
    raise ValueError("Elitism, Mutation, and Crossover rates must sum to 1")

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
    if pop[0][1] == MAX_FITNESS:
        print(f"Relative accuracy achieved at generation {g} with chromosome {str(pop[0][0])}")
        break
    new_pop = []

    # Elitism
    enum = int(ELITISM_RATE * POPULATION_SIZE)
    for i in range(enum):
        new_pop.append(pop[i])

    # Crossover
    while len(new_pop) < POPULATION_SIZE:
        p1 = tournament_selection(random.sample(pop, TOURNAMENT_SIZE)) # Tournament selection on k random chromosomes
        p2 = tournament_selection(random.sample(pop, TOURNAMENT_SIZE)) # Select 5 random chromosomes and pick the best one as parent 2
        if random.random() < CROSSOVER_RATE:
            c1, c2 = crossover(p1, p2)
        else:
            c1, c2 = p1, p2
        new_pop.append((c1, 0))
        new_pop.append((c2, 0))
    
    if len(new_pop) > POPULATION_SIZE:
        new_pop.pop() # Remove excess chromosome if we went over population size

    # Mutate
    for i in range(enum, len(new_pop)):
        if random.random() < MUTATION_RATE:
            new_pop[i] = (mutation(new_pop[i][0]), 0)
    
    # Update pop
    pop = new_pop