
from utils.evolution_functions import generate_random_samples, next_generation
from utils.plot_utils import plot_all
from utils.statistic_utils import run_chi_squared_test

def run_experiment(u_list, v_list, p0_list, q0_list, N, t, n, precision=10):
    estag_gens = {}
    eq_gens = {}
    gen_loss = {}
    ehw_gens = {}

    # EHW é P0 e Q0, por isso, precisamos comparar os valores obtidos com os valores iniciais
    # Utilizando o chi squared

    for u in u_list:
        for v in v_list:
            p_t_list = []
            q_t_list = []
            p2_t_list = []
            q2_t_list = []
            pq_t_list = []
            chi_squared_list = []
            for k, p0 in enumerate(p0_list):
                q0 = q0_list[k]

                p_t = [p0]
                q_t = [q0]
                p2_t = [p0 ** 2]
                q2_t = [q0 ** 2]
                pq_t = [2 * p0 * q0]
                chi_squared = [0]

                expected = generate_random_samples(p0, q0, N)
                dict_key = f"u={u}, v={v}, Pop={p0:.4f}x{q0:.4f}"
                print(f" p inicial, {p0}, q inicial, {q0}")
                for i in t:
                    if i == 0:
                        continue

                    p, q = next_generation(p_t[i-1], q_t[i-1], u, v)

                    if round(p, precision) == round(q, precision) and not eq_gens.get(dict_key, None):
                        # print(f"Gen {i} p: {p} q: {q}")
                        eq_gens[dict_key] = i
                    
                    if round(p, precision) == round(p_t[i-1], precision) and not estag_gens.get(dict_key, None):
                        print(f"Estagnation in Gen {i}\n p: {p} q: {q}")
                        estag_gens[dict_key] = i
                    
                    p2 = p ** 2
                    q2 = q ** 2
                    pq = 2 * p * q

                    # ehw = p**2 + 2*p*q + (q)**2
                    # if ehw != 1:
                    #     print(f"Gen {i} Não em EHW")
                    
                    observed = generate_random_samples(p, q, N)
                    chi_2, is_correlated = run_chi_squared_test(observed, expected, n)
                    if not is_correlated and not ehw_gens.get(dict_key, None):
                        print(f"Not in EHW in gen {i}")
                        ehw_gens[dict_key] = i

                    # Verify genetic loss
                    if int(pq * N) == 0 and not gen_loss.get(dict_key, None):
                        print(f"Genetic Loss in Gen {i}")# p: {p} q: {q}")
                        gen_loss[dict_key] = i
                    
                    # Add chi squared

                    # print(f"Gen {i} p: {p} q: {q}")
                    p_t.append(p)
                    q_t.append(q)
                    p2_t.append(p2)
                    q2_t.append(q2)
                    pq_t.append(pq)
                    chi_squared.append(chi_2)
                
                p_t_list.append(p_t)
                q_t_list.append(q_t)
                p2_t_list.append(p2_t)
                q2_t_list.append(q2_t)
                pq_t_list.append(pq_t)
                chi_squared_list.append(chi_squared)


                print(f"Final p: {p_t[-1]}, Final q: {q_t[-1]}")

                plot_all(t, [p_t], [q_t], [p2_t], [q2_t], [pq_t], [chi_squared], [p0], [q0], u, v, gen_ehw=ehw_gens.get(dict_key, None))
            # plot_n_pt(t, p_t_list, p0_list, u, v)

            plot_all(t, p_t_list, q_t_list, p2_t_list, q2_t_list, pq_t_list, chi_squared_list, p0_list, q0_list, u, v, plot_title=False)
            # For each p value do the plot in a subplot
            # plot_n_pt_qt(t, p_t_list, q_t_list, p0_list, q0_list, u, v)
            # plot_n_p2_q2_pq(t, p2_t_list, q2_t_list, pq_t_list, p0_list, q0_list, u, v)

            # plot_pt(t, p_t, p0)
            # plot_pt(t, q_t, q0)

            # plot_pt_qt(t, p_t, q_t, p0, q0)

            if estag_gens:
                # print("Estagnação")
                for key, value in estag_gens.items():
                    print(f"Estag Gen {key}: {value}")

            if eq_gens:
                #print("Equilibrio p=q")
                for key, value in eq_gens.items():
                    print(f"Eq Gen {key}: {value}")

            if ehw_gens:
                # print("Fim do EHW")
                for key, value in ehw_gens.items():
                    print(f"End of EHW ({key}) Gen = {value}")

            if gen_loss:
                # print("Perda Genética")
                for key, value in gen_loss.items():
                    print(f"Gen Loss {key}: {value}")