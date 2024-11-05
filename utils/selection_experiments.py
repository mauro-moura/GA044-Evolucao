
import numpy as np
import matplotlib.pyplot as plt

from utils.evolution_functions import generate_random_samples
from utils.statistic_utils import run_chi_squared_test

def next_gen(p2, pq, q2, wAA, wAa, waa):
    # Calcular frequência ajustada
    wAA_adj = wAA * p2
    wAa_adj = wAa * pq
    waa_adj = waa * q2

    # Calcular aptidão média
    w_med = wAA_adj + wAa_adj + waa_adj

    wAA_apos = wAA_adj / w_med
    wAa_apos = wAa_adj / w_med
    waa_apos = waa_adj / w_med

    # Calcular frequência após
    p1 = wAA_apos + 0.5 * wAa_apos
    q1 = 1 - p1

    return p1, q1, w_med

def calc_freq(p0, q0):
    p2 = p0**2
    pq = 2*p0*q0
    q2 = q0**2
    return p2, pq, q2

def run(p0, q0, wAA, wAa, waa, n_gen):
    decimals = 10
    N = 1e9
    n = 1
    p2, pq, q2 = calc_freq(p0, q0)

    p_list = []
    q_list = []
    p2_list = []
    pq_list = []
    q2_list = []
    delta_p = []
    w_med_list = []

    p_list.append(p0)
    q_list.append(q0)
    p2_list.append(p2)
    pq_list.append(pq)
    q2_list.append(q2)

    expected = generate_random_samples(p0, q0, N)

    ehw_gens = {}
    dict_key = f'{wAA}_{wAa}_{waa}'
    for i in range(n_gen):
        p, q, w_med = next_gen(p2_list[i], pq_list[i], q2_list[i], wAA, wAa, waa)
        p2, pq, q2 = calc_freq(p, q)

        # print(w_med)
        # print(p, q)
        # print(p2, pq, q2)

        if round(p, decimals) == round(p_list[i], decimals):
            print(f'Equilíbrio atingido em {i}.')
            break

        observed = generate_random_samples(p, q, N)
        chi_2, is_correlated = run_chi_squared_test(observed, expected, n)
        if not is_correlated and not ehw_gens.get(dict_key, None):
            ehw_gens[dict_key] = i

        delta_p.append(abs(p - p_list[i]))
        p_list.append(p)
        q_list.append(q)
        p2_list.append(p2)
        pq_list.append(pq)
        q2_list.append(q2)
        w_med_list.append(w_med)

    print(delta_p)
    # No plot, marcar quando sai do EHW...

    # Plot da paisagem adaptativa
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.plot(p_list, label='p')
    plt.plot(q_list, label='q')
    if ehw_gens.get(dict_key, None): plt.axvline(x=ehw_gens.get(dict_key, None), color='r', linestyle='--', label='EHW')
    plt.title("P e Q por T")
    plt.legend()


    plt.subplot(2, 3, 2)
    plt.plot(p2_list, label='p2')
    plt.plot(pq_list, label='pq')
    plt.plot(q2_list, label='q2')
    plt.title("P², PQ e Q² por T")
    if ehw_gens.get(dict_key, None): plt.axvline(x=ehw_gens.get(dict_key, None), color='r', linestyle='--', label='EHW')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(w_med_list, label='w_med')
    plt.title("w médio por T")
    if ehw_gens.get(dict_key, None): plt.axvline(x=ehw_gens.get(dict_key, None), color='r', linestyle='--', label='EHW')
    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.plot(p_list, p2_list, label='p x AA')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(p_list, pq_list, label='p x Aa')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(p_list, q2_list, label='p x aa')
    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 10))
    plt.plot(w_med_list, p_list[1:], label='w_med x p')
    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 10))
    plt.plot(delta_p, label='delta_p x T')
    plt.legend()
    plt.show()

def run_experiment_with_population(p0, q0, wAA, wAa, waa, pop_size, n_gen):
    decimals = 10
    n_gen_eq_count = 0
    n = 1
    p2, pq, q2 = calc_freq(p0, q0)

    p_list = []
    q_list = []
    p2_list = []
    pq_list = []
    q2_list = []
    delta_p = []
    w_med_list = []

    p_list.append(p0)
    q_list.append(q0)
    p2_list.append(p2)
    pq_list.append(pq)
    q2_list.append(q2)

    expected = generate_random_samples(p0, q0, pop_size)

    ehw_gens = {}
    dict_key = f'{wAA}_{wAa}_{waa}'
    for i in range(n_gen):
        p, q, w_med = next_gen(p2_list[i], pq_list[i], q2_list[i], wAA, wAa, waa)
        p2, pq, q2 = calc_freq(p, q)

        # print(w_med)
        # print(p, q)
        # print(p2, pq, q2)

        if int(p * pop_size) == int(p_list[i] * pop_size):
            if n_gen_eq_count == 0: print(f'Equilíbrio atingido em {i + 1}.')
            n_gen_eq_count += 1
        
        if n_gen_eq_count > 20:
            print("Equilíbrio atingido por 20 gerações. Parando...")
            break

        observed = generate_random_samples(p, q, pop_size)
        chi_2, is_correlated = run_chi_squared_test(observed, expected, n)
        if not is_correlated and not ehw_gens.get(dict_key, None):
            ehw_gens[dict_key] = i

        delta_p.append(abs(p - p_list[i]))
        p_list.append(p)
        q_list.append(q)
        p2_list.append(p2)
        pq_list.append(pq)
        q2_list.append(q2)
        w_med_list.append(w_med)

    print(delta_p)
    # No plot, marcar quando sai do EHW...

    # Plot da paisagem adaptativa
    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.plot(p_list, label='p')
    plt.plot(q_list, label='q')
    if ehw_gens.get(dict_key, None): plt.axvline(x=ehw_gens.get(dict_key, None), color='r', linestyle='--', label='EHW')
    plt.title("P e Q por T")
    plt.legend()


    plt.subplot(2, 3, 2)
    plt.plot(p2_list, label='p2')
    plt.plot(pq_list, label='pq')
    plt.plot(q2_list, label='q2')
    plt.title("P², PQ e Q² por T")
    if ehw_gens.get(dict_key, None): plt.axvline(x=ehw_gens.get(dict_key, None), color='r', linestyle='--', label='EHW')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(w_med_list, label='w_med')
    plt.title("w médio por T")
    if ehw_gens.get(dict_key, None): plt.axvline(x=ehw_gens.get(dict_key, None), color='r', linestyle='--', label='EHW')
    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    plt.plot(p_list, p2_list, label='p x AA')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(p_list, pq_list, label='p x Aa')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(p_list, q2_list, label='p x aa')
    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 10))
    plt.plot(w_med_list, p_list[1:], label='w_med x p')
    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 10))
    plt.plot(delta_p, label='delta_p x T')
    plt.legend()
    plt.show()
