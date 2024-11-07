
import numpy as np
import random

from numba import jit, prange
import matplotlib.pyplot as plt

from utils.genetic_drift_utils import genetic_drift

def mutation_p(p0, u, t):
    p = p0 * (1 - u)**t
    return p

def mutation_q(p0, v, t):
    q = p0 * (1 - v)**t
    return q

def generate_random_samples(p, q, N):
    values = [p**2 * N, 2*p*q*N, q**2 * N]
    return values

def equilibrium_frequency(u, v):
    """
    Given the mutation rates u and v, returns the equilibrium frequency of alleles p and q
    Source: Apostila do CEDERG.
    """
    p = v / (u + v)
    q = u / (u + v)
    return p, q

def next_generation(p0, q0, u, v):
    '''
    p0, q0 = População em t-1 (não população inicial)

    u, v = taxas de mutação
    '''
    p = p0 * (1 - u) + q0 * v
    q = p0 * u + q0 * (1 - v)
    return p, q

# def next_finite_generation(p0, q0, u, v, pop_size, seed=2024):
#     '''
#     p0, q0 = População em t-1 (não população inicial)

#     u, v = taxas de mutação
#     '''
#     random.seed(seed)
#     np.random.seed(seed)
    
#     population = np.array(["p"] * int(p0 * pop_size) + ["q"] * (pop_size - int(p0 * pop_size)))
#     np.random.shuffle(population)
    
#     for i in range(pop_size):
#         if population[i] == "p" and random.random() < u:
#             population[i] = "q"
#         elif population[i] == "q" and random.random() < v:
#             population[i] = "p"

#     p = np.sum(population == "p") / pop_size
#     q = 1 - p
    
#     return p, q

@jit(nopython=True, parallel=True)
def mutate_population(population, u, v):
    for i in prange(len(population)):
        if population[i] == "p" and random.random() < u:
            population[i] = "q"
        elif population[i] == "q" and random.random() < v:
            population[i] = "p"
    return population

def next_finite_generation(p0, q0, u, v, pop_size, seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    
    population = np.array(["p"] * int(p0 * pop_size) + ["q"] * int(q0 * pop_size))
    np.random.shuffle(population)
    
    population = mutate_population(population, u, v)
    
    p = np.sum(population == "p") / pop_size
    q = 1 - p
    
    return p, q

def next_finite_drift_generation(p0, q0, u, v, pop_size, seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    
    population = np.array(["p"] * int(p0 * pop_size) + ["q"] * int(q0 * pop_size))
    np.random.shuffle(population)
    
    population = genetic_drift(population)
    population = mutate_population(population, u, v)
    
    p = np.sum(population == "p") / pop_size
    q = 1 - p
    
    return p, q