
import random
from joblib import Parallel, delayed

# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
alphabet = ['A', 'T', 'G', 'C']

def generate_random_phrase(size, alphabet):
    alphabet_size = len(alphabet)
    
    return ''.join([alphabet[random.randint(0, alphabet_size - 1)] for i in range(size)])

def generate_random_phrase_with_mutation(initial_phrase, alphabet, u):
    new_phrase = list(initial_phrase)
    for i in range(len(new_phrase)):
        # chance = random.randint(0, 10**5)/ 10**5
        chance = random.random()
        if u <= chance:
            new_phrase[i] = random.choice(alphabet)
    return ''.join(new_phrase)