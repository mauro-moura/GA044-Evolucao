
import matplotlib.pyplot as plt

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

def plot_all(t, p_t_list, q_t_list, p2_t_list, q2_t_list, pq_t_list, chi_squared_list, p0_list, q0_list, u, v, chi_critic=3.84, gen_ehw=None, plot_title=True):
    plt.figure(figsize=(16, 9))

    plt.subplot(3, 3, 1)
    counter = 0
    for p_t, p0, q_t, q0 in zip(p_t_list, p0_list, q_t_list, q0_list):
        if counter > 1:
            plt.plot(t, p_t, label=f'p_{counter} = {p0:.4f}')
            plt.plot(t, q_t, label=f'q_{counter} = {q0:.4f}')
        else:
            plt.plot(t, p_t, label=f'p_0 = {p0:.4f}')
            plt.plot(t, q_t, label=f'q_0 = {q0:.4f}')
        if gen_ehw:
            plt.axvline(x=gen_ehw, color='r', linestyle='--')
        counter += 1
    plt.xlabel('Generation')
    plt.ylabel('p(t), q(t)')
    plt.legend()

    plt.subplot(3, 3, 2)
    counter = 0
    for p2_t, q2_t, pq_t in zip(p2_t_list, q2_t_list, pq_t_list):
        if counter > 1:
            plt.plot(t, p2_t, label=f'p²_{counter}')
            plt.plot(t, q2_t, label=f'q²_{counter}')
            plt.plot(t, pq_t, label=f'pq_{counter}')
        else:
            plt.plot(t, p2_t, label=f'p²')
            plt.plot(t, q2_t, label=f'q²')
            plt.plot(t, pq_t, label=f'pq')
        if gen_ehw:
            plt.axvline(x=gen_ehw, color='r', linestyle='--')
        counter += 1
    plt.xlabel('Generation')
    plt.ylabel('Frequences')
    plt.legend()

    plt.subplot(3, 3, 3)
    counter = 0
    for chi_squared in chi_squared_list:
        if counter > 1:
            plt.plot(t, chi_squared, label=f'chi²_{counter}')
        else:
            plt.plot(t, chi_squared, label=f'chi²')
        # Draw a red line on the critical value
        plt.axhline(y=chi_critic, color='r', linestyle='--')
        counter += 1
    plt.xlabel('Generation')
    plt.ylabel('chi squared')
    plt.legend()

    if plot_title:
        plt.suptitle(f'u = {u}, v = {v}')
    else:
        plt.suptitle("All plots merge")
    plt.show()
