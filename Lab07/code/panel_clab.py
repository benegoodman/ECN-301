# ---------------------------------------------------------
#    panel_clab.py
#
#    Code for panel data analysis
#    A quick summary of key estimators and tests
#    Using the Phillipine rice data
#
#    Olvar Bergland, March 2021/2022
#

#
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import linearmodels as plm
#       import sys


#
# load data
rice = pd.read_csv('./rice3.csv')
print(rice.info())

#
# create new variables
rice['lnQ'] = np.log(rice['prod'])
rice['lnD'] = np.log(rice['area'])
rice['lnL'] = np.log(rice['labor'])
rice['lnF'] = np.log(rice['fert'])
rice['lnE'] = np.log(rice['educ'])


#
# this is how to calculate summary statistics
# for each member of a group
#

#
# create a variable of means for each farmer, and merge
rice = rice.set_index('farmid')
rice['lnD_b'] = rice.groupby(['farmid'])[['lnD']].mean()
rice['lnL_b'] = rice.groupby(['farmid'])[['lnL']].mean()
rice['lnF_b'] = rice.groupby(['farmid'])[['lnF']].mean()
rice = rice.reset_index()
#


#
# create dummy variables
ydummies = pd.get_dummies(rice['year'],prefix='yd',drop_first=True)
rice = pd.concat([rice,ydummies], axis=1)

#
# also want a
year = pd.Categorical(rice.year)

#
# count individual and periods
N = rice['farmid'].nunique()
T = rice['year'].nunique()
print('Cross-sectional units : {}'.format(N))
print('Number of time periods: {}'.format(T))

rice = rice.set_index(['farmid','year'])
rice['year'] = year
print(rice.info())


#
# pooled OLS I
#
pom = plm.PooledOLS.from_formula(
           formula='lnQ ~ 1 + lnD + lnL + lnF + yd_1991 + yd_1992 + yd_1993 + yd_1994 + yd_1995 + yd_1996 + yd_1997',
           data=rice)
por = pom.fit(cov_type='clustered', cluster_entity=True)
print(por)
#
# test for period effects
yhyp = ['yd_1991=0', 'yd_1992=0', 'yd_1993=0', 'yd_1994=0', 'yd_1995=0', 'yd_1996=0', 'yd_1997=0']
wtest = por.wald_test(formula=yhyp)
#print(wtest)
print('Testing year effect in POLS')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))
print()

#
# pooled OLS II
#
pom = plm.PooledOLS.from_formula(
           formula='lnQ ~ 1 + lnD + lnL + lnF + year',
           data=rice)
por = pom.fit(cov_type='clustered', cluster_entity=True)
print(por)
#
# test for period effects
yhyp = ['year[T.1991]=0', 'year[T.1992]=0', 'year[T.1993]=0', 'year[T.1994]=0', 'year[T.1995]=0', 'year[T.1996]=0', 'year[T.1997]=0']
wtest = por.wald_test(formula=yhyp)
#print(wtest)
print('Testing year effect in POLS')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))
print()


#
# preliminary test for unobserved effects
#
rice['rhat'] = por.resids
rice['lrhat'] = rice.rhat.shift()
pmd = plm.PooledOLS.from_formula(
            formula='lnQ ~ 1 + lnD + lnL + lnF + year + lrhat',
            data=rice)
pmr = pmd.fit(cov_type='clustered', cluster_entity=True)

uhyp = ['lrhat=0']
wtest = pmr.wald_test(formula=uhyp)
#print(wtest)
print('Testing unobserved effects')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))


#
# fixed effects estimator
#
fem = plm.PanelOLS.from_formula(
            formula='lnQ ~ 1 + lnD + lnL + lnF + year + lrhat + EntityEffects',
            data=rice)
fer = fem.fit(cov_type='clustered', cluster_entity=True)
print(fer)


#
# random effects estimator
#
rem = plm.RandomEffects.from_formula(
    formula='lnQ ~ 1 + lnD + lnL + lnF + year + lrhat + EntityEffects',
    data=rice)
rer = rem.fit(cov_type='clustered', cluster_entity=True)
print(rer)


#
# correlated random effects estimator
#
crm = plm.RandomEffects.from_formula(
    formula='lnQ ~ 1 + lnD + lnL + lnF + lnD_b + lnL_b + lnF_b + year + lrhat + EntityEffects',
    data=rice)
crr = crm.fit(cov_type='clustered', cluster_entity=True)
print(crr)

#
# testing RE vs FE
#
uhyp = ['lnD_b=0', 'lnL_b=0', 'lnF_b=0']
wtest = crr.wald_test(formula=uhyp)
print('Testing FE vs RE')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))

#
# testing constant returns to scale
#
uhyp = ['lnD + lnL + lnF = 1']
wtest = crr.wald_test(formula=uhyp)
print('Testing CRS in CD')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))


#
# comparing results
print(plm.panel.compare({'POLS': por,
                         'FE'  : fer,
                         'RE'  : rer,
                         'CRE' : crr},
precision='std_errors'))

