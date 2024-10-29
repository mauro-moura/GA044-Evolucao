import matplotlib.pyplot as plt

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
