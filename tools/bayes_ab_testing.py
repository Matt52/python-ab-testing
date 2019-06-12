import numpy as np

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

