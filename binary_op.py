import numpy as np
from numpy.random import randint, rand

# tournament selection
def selection(pop, scores, k=3):
    """return one selected parent after run the tournament,
    the tournament consist on chossing k-1 samples of the population
    and keeps the one with the best score"""
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
    # check if better (e.g. perform a tournament) "minimize"
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    """Cross the parents given a r_cross probability, if not,
    returns a copy of the parents"""
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = np.append(p1[:pt], p2[pt:])
        c2 = np.append(p2[:pt], p1[pt:])
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    """Performs mutation given a r_mut probability, it just flips
    a bit if rand is higher than the probability"""
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
    return bitstring

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    """performs a genetic optimization"""
    pop = [randint(0, 2, n_bits) for _ in range(n_pop)]

    # keep track of best solution
    best, best_eval = 0, objective(pop[0])

    for gen in range(n_iter):
        scores = [objective(c) for c in pop]
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        
        #select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]

        children = list()

        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]

            #crossover and mutation
            for c in crossover(p1, p2, r_cross):
                #mutation
                mut_child = mutation(c, r_mut)

                #store for next generation
                children.append(mut_child)

                #replace population
                pop = children
    return [best, best_eval]

def onemax(x):
    "function to minimize/maximize"
    return -sum(x)

# define the total iterations
n_iter = 100
# bits
n_bits = 10
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)

# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))