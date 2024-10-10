
# Chi squared test
def chi_squared_test(observed, expected):
    chi_squared = 0
    for i in range(len(observed)):
        chi_squared += ((observed[i] - expected[i]) ** 2) / expected[i]
    return chi_squared
