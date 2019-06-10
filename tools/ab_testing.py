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


def z_test_sample_size(conv_r, mde, confidance = 0.95, power = 0.8, test_share_size = 0.5):
    alpha = 1 - confidance
    conv_exp = conv_r * (1 + mde)
    var = conv_r * (1 - conv_r) + conv_exp * (1 - conv_exp)
    
    size = (var / ((conv_exp - conv_r)**2)) * (norm.ppf(power) + norm.ppf(1 - alpha/2))**2

    share_ratio = (1 - test_share_size) / test_share_size
    size_adj = (2*size*((1 + share_ratio)**2)) / (4*share_ratio)

    return [(size_adj) / (1 + share_ratio), (size_adj*share_ratio) / (1 + share_ratio)]


#version that should be used in abtestguide to be consistent with their power calculation
def z_test_sample_size_alternative(conv_r, mde, confidance = 0.95, power = 0.8):
    alpha = 1 - confidance
    conv_exp = conv_r * (mde + 1)

    size = ((norm.ppf(1 - alpha/2) * np.sqrt(conv_r * (1 - conv_r)) + norm.ppf(power) * np.sqrt(conv_exp * (1 - conv_exp))) / (conv_exp - conv_r))**2

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
p-value: {8}
power: {9}""".format(winning, result, round(mu_A*100,3), round(mu_B*100,3), round(uplift*100,3), round(se_A,5), round(se_B,5), round(Z,5), round(pvalue,5), round(power,5))
         )

##################################################################################################################

def pbb_conversion(totals, successes, sim_count = 10000):
    beta_samples = np.array([np.random.beta(successes[i] + 1/2, totals[i] - successes[i] + 1/2, sim_count) for i in range(len(totals))])
    
    max_values = np.argmax(beta_samples, axis=0)
    unique, counts = np.unique(max_values, return_counts=True)
    ocurrences = dict(zip(unique, counts))
    
    result = []
    for i in range(len(totals)):
        result.append(round(ocurrences.get(i, 0) / sim_count, 5))
    
    return result


##insert agg data - total successes, sum of log(revenue), sum of log^2(revenue)
def lognormal_posteriors(successes, log_revenues, log_2_revenues, sim_count = 10000, m = 1, a = 0, b = 0, w = 0.01):
    if successes <= 0:
        return np.zeros(sim_count)
    else:
        #we assume that logarithms of revenue are normaly distributed
        x_bar = (log_revenues / successes)
        a_post = a+(successes / 2)
        b_post = b+(1/2)*(log_2_revenues - 2*log_revenues*x_bar + successes*(x_bar**2))\
                 + ((successes*w)/(2 * (successes + w)))*((x_bar-m)**2)

        sig_2 = (1 / np.random.gamma(a_post, 1 / b_post, sim_count)) #has to be 1/b - it is a scale, not a rate

        m_post = ((successes*x_bar + w*m)/(successes+w))
        sig_2_post = ((sig_2)/(successes + w))

        normal_post = np.random.normal(m_post, np.sqrt(sig_2_post))
    
        return np.exp(normal_post + (sig_2/2))



def pbb_revenue(totals, successes, log_revenues, log_2_revenues, sim_count = 10000, m = 1, a = 0, b = 0, w = 0.01):
    
    if max(successes) <= 0:
        #if no success 
        return np.full(len(totals), 1 / len(totals))
    else:
        beta_samples = np.array(
            [np.random.beta(successes[i] + 1/2, totals[i] - successes[i] + 1/2, sim_count)
             for i in range(len(totals))
            ]
        )

        lognormal_samples = np.array(
            [
                lognormal_posteriors(successes[i], log_revenues[i], log_2_revenues[i], sim_count, m, a, b, w)
                for i in range(len(totals))
            ])

        revenue_samples = beta_samples * lognormal_samples
        
        max_values = np.argmax(revenue_samples, axis=0)
        unique, counts = np.unique(max_values, return_counts=True)
        ocurrences = dict(zip(unique, counts))
        
        result = []
        for i in range(len(totals)):
            result.append(round(ocurrences.get(i, 0) / sim_count, 5))
        
        return result


