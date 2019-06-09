import numpy as np
from scipy.stats import norm


def AB_test(variants, totals, successes, confidance = 0.95, sides = 2):
    
    nA = totals[0]
    cA = successes[0]
    nB = totals[1]
    cB = successes[1]
    alpha = 1 - confidance
    mu_A = cA / nA
    mu_B = cB / nB
    
    if mu_A > mu_B:
        winning = variants[0]
    elif mu_A < mu_B:
        winning = variants[1]
    else:
        winning = 'None'
    
    uplift = (mu_B - mu_A) / mu_A
    var_A = mu_A * (1 - mu_A)
    var_B = mu_B * (1 - mu_B)
    se_A = np.sqrt(var_A / nA)
    se_B = np.sqrt(var_B / nB)
    
    Z = (mu_B - mu_A) / np.sqrt(se_A**2 + se_B**2)
    pvalue = sides * (1 - norm.cdf(abs(Z)))
    
    if pvalue < (1 - confidance):
        result = 'significant'
    else:
        result = 'not significant'
    
    if mu_A <= mu_B:
        x = mu_A + norm.ppf(1 - (alpha / sides)) * se_A #i.e. ppf = 1.96 for alpha = 0.05 and sides = 2
        power = 1 - norm.cdf((x - mu_B) / se_B)
    else:
        x = mu_B + norm.ppf(1 - (alpha / sides)) * se_B
        power = 1 - norm.cdf((x - mu_A) / se_A)
        
    
    print("""winning: {0} ({1} - confidance: {10})
conversion rate A: {2}%
conversion rate B: {3}%
uplift: {4}%
se A: {5}
se B: {6}
Z-score: {7}
p-value: {8}
power: {9}""".format(winning, result, round(mu_A*100,3), round(mu_B*100,3), round(uplift*100,3), round(se_A,5), round(se_B,5), round(Z,5), round(pvalue,5), round(power,5), confidance)
         )

#########################################################

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


