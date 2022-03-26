# ---------------------------------------------------------
#    panel_estimators.py
#
#    Code for panel data analysis
#    A quick summary of the key estimators
#
#    Olvar Bergland, March 2021/2022
#

#
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
airf.info()

#
# count individual and periods
N = airf['id'].nunique()
T = airf['year'].nunique()
print('Cross-sectional units : {}'.format(N))
print('Number of time periods: {}'.format(T))

#
# I need an extra variable referencing the year
airf['t'] = airf.year

#
# create a variable of means for each airport, and merge
#  this is accomplished by having a proper index on entity
airf = airf.set_index(['id'])
airf['concen_b'] = airf.groupby(['id'])[['concen']].mean()
airf = airf.reset_index()


#
# define a proper index for panel data
#
airf = airf.set_index(['id','year'])



#
#
# all the different estimation models for panel data
#
#

#
# pooled OLS
#
pom = plm.PooledOLS.from_formula(
    formula='lfare ~ 1 + concen + ldist + ldistsq +  C(t)',
    data=airf)
por = pom.fit(cov_type='clustered', cluster_entity=True)


#
# first difference estimator
#
pmd = plm.FirstDifferenceOLS.from_formula(formula='lfare ~ concen',
                                          data=airf)
pmc = pmd.fit(cov_type='clustered', cluster_entity=True)

#
# first difference estimator
#
fdm = plm.FirstDifferenceOLS.from_formula(
    formula='lfare ~ concen + y98 + y99 + y00',
    data=airf)
fdr = fdm.fit(cov_type='clustered', cluster_entity=True)


#
# fixed effects estimator
#
fem = plm.PanelOLS.from_formula(
    formula='lfare ~ 1 + concen + EntityEffects + C(t)',
    data=airf)
fer = fem.fit(cov_type='clustered', cluster_entity=True)


#
# random effects estimator
#
rem = plm.RandomEffects.from_formula(
    formula='lfare ~ 1 + concen + ldist + ldistsq + C(t) + EntityEffects',
    data=airf)
rer = rem.fit(cov_type='clustered', cluster_entity=True)


#
# correlated random effects estimator
#
crm = plm.RandomEffects.from_formula(
    formula='lfare ~ 1 + concen + concen_b + ldist + ldistsq + C(t) + EntityEffects',
    data=airf)
crr = crm.fit(cov_type='clustered', cluster_entity=True)

rehyp = ['concen_b=0']
wtest = crr.wald_test(formula=rehyp)
print(wtest)
print('Testing for correlated effects')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))

#
# comparing results
print(plm.panel.compare({'POLS': por, 'FD': fdr, 'FE': fer, 'RE': rer, 'CRE': crr},
                        precision='std_errors'))

