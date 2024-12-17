# %%
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
alphabet = ['A', 'T', 'G', 'C']

# %%
class BaseModel():
    def __init__(self):
        self.y_true = np.empty([])
        self.y_pred = np.empty([])
    
    def initialize_population(self, population_size:int, phrase_length:int):
        """
        Initializes the population with random values
        """

        def generate_random_phrase(size, alphabet):
            # alphabet_size = len(alphabet)
            # return ''.join([alphabet[random.randint(0, alphabet_size - 1)] for i in range(size)])
            return ''.join(np.random.choice(alphabet, size))

        self.y_true = np.array([generate_random_phrase(phrase_length, alphabet) for _ in range(population_size)])
        self.y_pred = np.array([generate_random_phrase(phrase_length, alphabet) for _ in range(population_size)])
    
    def fitness_function(self, y_true:str, y_pred:str):
        count = 0
        for i, letter in enumerate(y_true):
            if letter == y_pred[i]:
                count += 1
        
        count = count / len(y_true)
        return count

    def evaluate_fitness(self):
        """
        Returns the fitness for all values in the y_true and y_pred arrays
        """
        return np.array([self.fitness_function(y_true, y_pred) for y_true, y_pred in zip(self.y_true, self.y_pred)])

# %%
def select_n_best(phrases: np.ndarray, n: int, base_model: BaseModel) -> np.ndarray:
    """
    Select n best phrases based on the fitness function
    and return the indexes of the selected phrases.
    """
    fitness = base_model.evaluate_fitness()
    return np.argsort(fitness)[-n:]

def get_worse_index(phrase, target, phrases, base_model: BaseModel):
    worse_score = base_model.fitness_function(target, phrase)
    for i in range(0, len(phrases)):
        score = base_model.fitness_function(target, phrases[i])
        if score < worse_score:
            return i
    return None

def genetic_drift(phrases: np.ndarray, seed: int = 2024) -> np.ndarray:
    """
    Apply genetic drift by randomly sampling phrases.
    """
    np.random.seed(seed)
    return np.random.choice(phrases, len(phrases), replace=True)

def crossover(population:np.array, base_model: BaseModel=None) -> np.array:
    '''
    Recombinação \n
    Pegar m% das frases, selecionar aleatoriamente 2 frases e trocar um pedaço delas dado um crossoverpoint aleatório
    '''
    phrase_size = len(population[0])
    crossover_point = np.random.randint(1, phrase_size - 1)
    # Select parents based on best fitness
    index_phrase1, index_phrase2 = select_n_best(population, 2, base_model)

    phrase1 = population[index_phrase1]
    phrase2 = population[index_phrase2]

    new_phrase_1 = phrase1[:crossover_point] + phrase2[crossover_point:]
    new_phrase_2 = phrase2[:crossover_point] + phrase1[crossover_point:]

    return new_phrase_1, new_phrase_2

def selection(new_phrases, phrase, base_model: BaseModel):
    return max(new_phrases, key=lambda p: base_model.fitness_function(phrase, p))

# %%
def plot_graphs_and_describe(full_gens, m_list):
    d = {}

    for i in range(len(full_gens)):
        d[f"M{m_list[i]}"] = full_gens[i]

    df = pd.DataFrame(d)

    print(df.describe())

    df.boxplot()
    plt.xticks(rotation=45)
    plt.ylabel("Generations")
    plt.xlabel("Mutation rate")
    plt.show()

# %%
def select_parents(population, fitness):
    probabilities = fitness - fitness.min() + 1e-6
    probabilities /= probabilities.sum()
    return np.random.choice(population, size=len(population), p=probabilities)

# %%
def mutate(individual, u):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < u:
            individual[i] = str(np.random.choice(alphabet))
    return ''.join(individual)

# %%
# 1st: 25.766667
phrase = 'GAGCCC'#GAACGAGCTTTGCGTCT'
n_phrases = 5
u = 0.25
m = 0.75
n_exp = 1
gen_max = 2
seed = 2024

random.seed(seed)
np.random.seed(seed)

base_model = BaseModel()

target = phrase
gens = []
for i in range(n_exp):
    base_model.initialize_population(n_phrases, len(phrase))
    fitness = base_model.evaluate_fitness()
    print(fitness)

    phrase_size = len(phrase)
    gen = 0
    initial_population = base_model.y_pred
    population = initial_population.copy()
    for gen in range(int(gen_max)):
        print(f"Generation {gen}")
        if phrase in population:
            break
        
        # Selection
        population = select_parents(population, fitness)
        for i in range(len(population) // 2):
            # Crossover
            new_phrase_1, new_phrase_2 = crossover(population, base_model)
            # population[i] = crossover(population, m)
            worse_index_1 = get_worse_index(new_phrase_1, target, population, base_model)
            worse_index_2 = get_worse_index(new_phrase_2, target, population, base_model)
            # Mutate
            if worse_index_1:
                population[worse_index_1] = mutate(new_phrase_1, u)
            if worse_index_2:
                population[worse_index_2] = mutate(new_phrase_2, u)
            
    gens.append(gen)

mean_gen = sum(gens) / len(gens)
print(f"Mean generations: {mean_gen:,}")

plot_graphs_and_describe([gens], [u])


