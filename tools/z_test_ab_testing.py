import numpy as np
from scipy.stats import norm


def z_test_power(mu_A, mu_B, se_A, se_B, alpha = 0.05, sides = 2):
    
    if mu_A <= mu_B:
        power = 1 - norm.cdf(norm.ppf(1 - (alpha / sides)) - (mu_B - mu_A)/np.sqrt(se_A**2 + se_B**2))
    else:
        power = 1 - norm.cdf(norm.ppf(1 - (alpha / sides)) - (mu_A - mu_B)/np.sqrt(se_A**2 + se_B**2))

    return power

#version used in abtestguide and surveymoneky calculators
def z_test_power_alternative(mu_A, mu_B, se_A, se_B, alpha = 0.05, sides = 2):
    if mu_A <= mu_B:
        x = mu_A + norm.ppf(1 - (alpha / sides)) * se_A 
        power = 1 - norm.cdf((x - mu_B) / se_B)
    else:
        x = mu_B + norm.ppf(1 - (alpha / sides)) * se_B
        power = 1 - norm.cdf((x - mu_A) / se_A)

    return power


def z_test_sample_size(conv_r, mde, confidance = 0.95, power = 0.8, test_share_size = 0.5, test_type = 'two-tailed'):

    if test_type not in ['one-tailed', 'two-tailed']:
        raise ValueError("Invalid test type. Expected one of: %s" % ['one-tailed', 'two-tailed'])
    elif test_type == 'one-tailed':
        sides = 1
    else:
        sides = 2

    alpha = 1 - confidance
    conv_exp = conv_r * (1 + mde)
    var = conv_r * (1 - conv_r) + conv_exp * (1 - conv_exp)
    
    size = (var / ((conv_exp - conv_r)**2)) * (norm.ppf(power) + norm.ppf(1 - alpha/sides))**2

    share_ratio = (1 - test_share_size) / test_share_size
    size_adj = (2*size*((1 + share_ratio)**2)) / (4*share_ratio)

    return [round((size_adj) / (1 + share_ratio),2), round((size_adj*share_ratio) / (1 + share_ratio),2)]


#version that should be used in abtestguide to be consistent with their power calculation
def z_test_sample_size_alternative(conv_r, mde, confidance = 0.95, power = 0.8, test_type = 'two-tailed'):
    if test_type not in ['one-tailed', 'two-tailed']:
        raise ValueError("Invalid test type. Expected one of: %s" % ['one-tailed', 'two-tailed'])
    elif test_type == 'one-tailed':
        sides = 1
    else:
        sides = 2

    alpha = 1 - confidance
    conv_exp = conv_r * (mde + 1)

    size = ((norm.ppf(1 - alpha/sides) * np.sqrt(conv_r * (1 - conv_r)) + norm.ppf(power) * np.sqrt(conv_exp * (1 - conv_exp))) / (conv_exp - conv_r))**2

    return size


def AB_z_test(totals_A, successes_A, totals_B, successes_B, confidance = 0.95, test_type = 'two-tailed'):
    
    if test_type not in ['one-tailed', 'two-tailed']:
        raise ValueError("Invalid test type. Expected one of: %s" % ['one-tailed', 'two-tailed'])
    elif test_type == 'one-tailed':
        sides = 1
    else:
        sides = 2

    alpha = 1 - confidance
    mu_A = successes_A / totals_A
    mu_B = successes_B / totals_B
    
    if mu_A > mu_B:
        winning = 'A'
    elif mu_A < mu_B:
        winning = 'B'
    else:
        winning = 'None'
    
    uplift = (mu_B - mu_A) / mu_A
    var_A = mu_A * (1 - mu_A)
    var_B = mu_B * (1 - mu_B)
    se_A = np.sqrt(var_A / totals_A)
    se_B = np.sqrt(var_B / totals_B)
    
    Z = (mu_B - mu_A) / np.sqrt(se_A**2 + se_B**2)
    pvalue = sides * (1 - norm.cdf(abs(Z)))
    
    if pvalue < (1 - confidance):
        result = 'significant'
    else:
        result = 'not significant'
    
    power = z_test_power(mu_A, mu_B, se_A, se_B, alpha, sides)
    #power = z_test_power_alternative(mu_A, mu_B, se_A, se_B, alpha, sides)
        
    
    print("""winning: {0} ({1})
conversion rate A: {2}%
conversion rate B: {3}%
uplift: {4}%
standard error A: {5}
standard error B: {6}
Z-score: {7}
Z-test p-value: {8}
Z-test power: {9}""".format(winning, result, round(mu_A*100,3), round(mu_B*100,3), round(uplift*100,3), round(se_A,5), round(se_B,5), round(Z,5), round(pvalue,5), round(power,5))
         )

