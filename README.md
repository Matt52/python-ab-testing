# python-ab-testing
### Python tools for AB testing (frequentist &amp; bayesian approach)

All functions could be found [here](tools).<br>
See [this notebook](AB_testing_demonstration.ipynb) for a demonstration.<br>

Calculation of z-test power may differ from some online calculators (see belove). I believe they have a different approach and to be honest I am not sure if it is correct. I opened a question [here](https://math.stackexchange.com/questions/3259058/ab-testing-power-of-2-sample-z-test) so maybe someone will answer it.<br>

Bayesian part is inspired mainly by [this Coursera course](https://www.coursera.org/learn/bayesian-statistics) which I can really recommend. Log-normal approach to revenue was inspired by [this blog post](https://www.richrelevance.com/blog/2013/08/26/bayesian-ab-testing-with-a-log-normal-model/). But be careful, there is a mistake in function _draw_mus_and_sigmas_ (_var_norm_ should be really equal to _sqrt(sig_sq_samples/kN)_).



**some online calculators:**<br>
https://www.surveymonkey.com/mp/ab-testing-significance-calculator/<br>
https://abtestguide.com/calc/<br>
https://abtestguide.com/abtestsize/<br>

**interesting reading:**<br>
https://towardsdatascience.com/the-art-of-a-b-testing-5a10c9bb70a4<br>
https://www.invespcro.com/blog/calculating-sample-size-for-an-ab-test/<br>
https://www.richrelevance.com/blog/2013/08/26/bayesian-ab-testing-with-a-log-normal-model/<br>
https://portal.pixelfederation.com/cs/blog/article/ab-testing-methodology-change<br>
http://varianceexplained.org/r/bayesian-ab-testing/<br>
https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce<br>