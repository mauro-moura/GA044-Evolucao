import matplotlib.pyplot as plt

def mutation_p(p0, u, t):
    p = p0 * (1 - u)**t
    return p

def mutation_q(p0, v, t):
    q = p0 * (1 - v)**t
    return q

def next_generation(p0, q0, u, v):
    p = p0 * (1 - u) + q0 * v
    q = p0 * u + q0 * (1 - v)
    return p, q

def plot_pt(t, p_t, p0):
    plt.plot(t, p_t, label=f'p = {p0:.4f}')
    plt.legend()
    # plt.title(f'u = {u}, v = {v}')
    plt.show()

def plot_n_pt(t, p_t_list, p0_list, u, v):
    for p_t, p0 in zip(p_t_list, p0_list):
        plt.plot(t, p_t, label=f'p = {p0:.4f}')
    plt.legend()
    plt.title(f'u = {u}, v = {v}')
    plt.show()

def plot_n_pt_qt(t, p_t_list, q_t_list, p0_list, q0_list, u, v):
    for p_t, q_t, p0, q0 in zip(p_t_list, q_t_list, p0_list, q0_list):
        plt.plot(t, p_t, label=f'p = {p0:.4f}')
        plt.plot(t, q_t, label=f'q = {q0:.4f}')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('p(t), q(t)')
    plt.title(f'u = {u}, v = {v}')
    plt.show()

def plot_n_p2_q2_pq(t, p2_t, q2_t, pq_t, p0_list, q0_list, u, v):
    counter = 0
    for p2_t, q2_t, pq_t, p0, q0 in zip(p2_t, q2_t, pq_t, p0_list, q0_list):
        counter += 1
        plt.plot(t, p2_t, label=f'p_{counter}')
        plt.plot(t, q2_t, label=f'q_{counter}')
        plt.plot(t, pq_t, label=f'pq_{counter}')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('p²(t), q²(t), pq(t)')
    plt.title(f'u = {u}, v = {v}')
    plt.show()

def plot_pt_qt(t, p_t, q_t, p0, q0):
    plt.plot(t, p_t, label=f'p = {p0:.4f}')
    plt.plot(t, q_t, label=f'q = {q0:.4f}')
    plt.legend()
    # plt.title(f'u = {u}, v = {v}')
    plt.show()

def plot_all(t, p_t_list, q_t_list, p2_t_list, q2_t_list, pq_t_list, chi_squared_list, p0_list, q0_list, u, v):
    plt.figure(figsize=(16, 9))

    plt.subplot(3, 3, 1)
    counter = 0
    for p_t, p0, q_t, q0 in zip(p_t_list, p0_list, q_t_list, q0_list):
        counter += 1
        plt.plot(t, p_t, label=f'p_{counter} = {p0:.4f}')
        plt.plot(t, q_t, label=f'q_{counter} = {q0:.4f}')
    plt.legend()

    plt.subplot(3, 3, 2)
    counter = 0
    for p2_t, q2_t, pq_t in zip(p2_t_list, q2_t_list, pq_t_list):
        counter += 1
        plt.plot(t, p2_t, label=f'p²_{counter}')
        plt.plot(t, q2_t, label=f'q²_{counter}')
        plt.plot(t, pq_t, label=f'pq_{counter}')
    plt.legend()

    plt.subplot(3, 3, 3)
    counter = 0
    for chi_squared in chi_squared_list:
        counter += 1
        plt.plot(t, chi_squared, label=f'chi²_{counter}')
    plt.legend()

    plt.suptitle(f'u = {u}, v = {v}')
    plt.show()