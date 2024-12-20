
import scipy.stats as stats

# Chi squared freedom table: https://towardsdatascience.com/statistics-in-python-using-chi-square-for-feature-selection-d44f467ca745
# In our case, we have 1 degrees of freedom (quantity of independent variables), so we will use the 0.05 cut value and the critical value of 3.84

# Chi squared test
def chi_squared_test(observed, expected):
    chi_squared = 0
    for i in range(len(observed)):
        chi_squared += ((observed[i] - expected[i]) ** 2) / expected[i]
    return chi_squared

def verify_chi_squared(chi_squared_value, degrees_of_freedom, cut_value=0.05):
    p_value = 1 - stats.chi2.cdf(chi_squared_value, degrees_of_freedom)
    # print(p_value)
    return p_value > cut_value # Se p_value < 0.05, não há EHW

def run_chi_squared_test(observed, expected, n=1):
    chi_squared = chi_squared_test(observed, expected)
    # is_correlated = chi_squared < 3.84
    is_correlated = verify_chi_squared(chi_squared, n)
    # print(f"Chi squared value: {chi_squared}, critical value: 3.84, is correlated: {is_correlated}")
    return chi_squared, is_correlated
