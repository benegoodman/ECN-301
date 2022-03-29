# ---------------------------------------------------------
#
#    Code for panel data analysis
#
#    Olvar Bergland, March 2021
#

#
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as plm

import wooldridge as woo

#
# use the airfare dataset from Wooldridge
#
airf = woo.dataWoo('airfare')
print(airf.info())

#
# summarize airline routes per year
print(airf['year'].value_counts())

#
# count individual routes and periods
N = airf['id'].nunique()
T = airf['year'].nunique()
print('Cross-sectional units : {}'.format(N))
print('Number of time periods: {}'.format(T))

#
# define a proper index based on id and year
# I create a new variable equal to year
airf['period'] = airf.year
airf = airf.set_index(['id','period'])

#
#
print(airf[['year','lfare','concen']].head())

#%%

#
# want dummy variables for the periods
#   - either create new variables
#   - or implicit in the model statement
#        coded as a contrast C(.)
#

#
# create dummy variables
ydummies = pd.get_dummies(airf['year'],prefix='yd',drop_first=True)
airf = pd.concat([airf,ydummies], axis=1)
print(airf.info())

#%%


"""
Example to show that we need to use clustered standard errors for pooled OLS estimation
"""

#
# pooled model, OLS estimation w/ dummy variables
pom = smf.ols(formula='lfare ~ concen + ldist + ldistsq + yd_1998 + yd_1999 + yd_2000',
              data=airf)
por = pom.fit(cov_type='HC3')
pot = pd.DataFrame({'b'   : round(por.params,5),
                    'se'  : round(por.bse, 5),
                    't'   : round(por.tvalues, 2),
                    'p'   : round(por.pvalues,3)})
print(pot)

#
# test for period effects
yhyp = ['yd_1998=0','yd_1999=0','yd_2000=0']
ftest = por.f_test(yhyp)
print('Testing year effect in POLS')
print('F-stat : {}'.format(ftest.statistic[0][0]))
print('p-value: {}'.format(ftest.pvalue))



#
# pooled model, OLS estimation w/ implicit coding of dummy variables
#
pom = smf.ols(formula='lfare ~ concen + ldist + ldistsq + C(year)',
              data=airf)
por = pom.fit(cov_type='HC3')
pot = pd.DataFrame({'b'   : round(por.params,5),
                    'se'  : round(por.bse, 5),
                    't'   : round(por.tvalues, 2),
                    'p'   : round(por.pvalues, 3)})
print(por.summary())

#
# test for period effects - need details about parameters
#   - find the names from the model results
#
yhyp = ['C(year)[T.1998]=0','C(year)[T.1999]=0','C(year)[T.2000]=0']
ftest = por.f_test(yhyp)
print('Testing year effect in POLS')
print('F-stat : {}'.format(ftest.statistic[0][0]))
print('p-value: {}'.format(ftest.pvalue))

"""
Conclusion: Time plays a significant part in estimation. 
"""


#%%
#


"""
Olvar uses pooled OLS to test for time varying, and individual effects - this is why we do these
"""


# POLS w/cluster robust standard errors
#
pom = plm.PooledOLS.from_formula(formula='lfare ~ 1 + concen + ldist + ldistsq + C(year)',
                                data=airf)
por = pom.fit(cov_type='robust')
print(por)
por = pom.fit(cov_type='clustered', cluster_entity=True)
print(por)

yhyp = ['C(year)[T.1998]=0','C(year)[T.1999]=0','C(year)[T.2000]=0']
wtest = por.wald_test(formula=yhyp)
print(wtest)
print('Testing year effect in POLS')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))

"""
Conclusion: use clustered standard errors for panel data when doing pooled OLS

"""

#%%

"""
lrhat = individual effects ...or rather all effects not accounted for by variables otherwise controlled for
"""

#
# preliminary test for unobserved effects
#
airf['rhat'] = por.resids
airf['lrhat'] = airf.rhat.shift()
pmd = plm.PooledOLS.from_formula(formula='lfare ~ 1 + concen + ldist + ldistsq + C(year) + lrhat',
                                 data=airf[airf['year']>1997])
pmc = pmd.fit(cov_type='clustered', cluster_entity=True)
print(pmc)

uhyp = ['lrhat=0']
wtest = pmc.wald_test(formula=uhyp)
print(wtest)
print('Testing unobserved effects')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))



#%%

"""
First diff: normally used in short panels (few time periods)
if T < 3 if yields the same estimators as FE estimators

Ben lambert has a video on it here: https://www.youtube.com/watch?v=G7WqK2o474Y

"""

#
# first difference estimator
#
fdm = plm.FirstDifferenceOLS.from_formula(formula='lfare ~ concen',
                                data=airf)
fdr = fdm.fit(cov_type='clustered', cluster_entity=True)
print(fdr)

"""
Yields biased results
"""

#%%

#
# add year dummy variables
fdm = plm.FirstDifferenceOLS.from_formula(formula='lfare ~ concen + yd_1999 + yd_2000',
                                          data=airf)
fdr = fdm.fit(cov_type='clustered', cluster_entity=True)
print(fdr)


#%%

"""
Note: EntityEffects need to be included, im at a loss at what it truly does. If you want 
the whole truth about it, ask Olvar. Short story: include it for technical reasons.

"""

#
# fixed effects estimator
#
fem = plm.PanelOLS.from_formula(
            formula='lfare ~ 1 + concen + C(year) + EntityEffects',
            data=airf)
fer = fem.fit(cov_type='clustered', cluster_entity=True)
print(fer)



#%%

#
# random effects estimator
#
rem = plm.RandomEffects.from_formula(
            formula='lfare ~ 1 + concen + ldist + ldistsq + C(year) + EntityEffects',
            data=airf)
rer = rem.fit(cov_type='clustered', cluster_entity=True)
print(rer)

#%%

#
# the tricky part with CRE is to create the
# entity specific averages
#

#
# this is how to calculate summary statistics
# for each member of a group
#
#print(airf.groupby(['year'])[['concen']].mean())

#
# create a variable of means for each airport, and merge
airf = airf.reset_index().set_index(['id'])
airf['concen_b'] = airf.groupby(['id'])[['concen']].mean()
airf = airf.reset_index()
#
# check result
print(airf[['id','year','concen','concen_b']])

#
# set a proper index
airf = airf.set_index(['id','period'])

#%%

"""
Note: a CRE estimator is basically the same as an RE estimator, but with average variables across entities

"""

#
# correlated random effects estimator
#
crm = plm.RandomEffects.from_formula(
            formula='lfare ~ 1 + concen + concen_b + ldist + ldistsq + C(year) + EntityEffects',
            data=airf)
crr = crm.fit(cov_type='clustered', cluster_entity=True)
print(crr)

#%%

#
# comparing results
print(plm.panel.compare({'POLS': por,
                         'FD'  : fdr,
                         'FE'  : fer,
                         'RE'  : rer,
                         'CRE' : crr},
                        precision='std_errors'))

"""
Conclusion:
    RE is inconsistent (i.e. time effects play a large part)
    We also see that coefficients are the same for FE and CRE models
        Means that differences between entities are not a source of systemic differences in outcome
    However, average concentration variable for CRE is significant 
    Hence, CRE model is correct (the intercept is off in RE, despite being more efficient)
    Had concen_b not been significant we would have gone with RE
"""
