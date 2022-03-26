# coding: utf-8

# In[1]:


#    Code for panel data analysis
#
#    Olvar Bergland, March 2021
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


# In[2]:


#
# summarize airline routes per year
print(airf['year'].value_counts())


# In[3]:


#
# count individual routes and periods
N = airf['id'].nunique()
T = airf['year'].nunique()
print('Cross-sectional units : {}'.format(N))
print('Number of time periods: {}'.format(T))


# In[4]:


#
# define a proper index
airf['t'] = airf.year
airf = airf.set_index(['id','year'])

#
#
print(airf[['lfare','concen']].head())


# In[5]:


#
# pooled model, OLS estimation
pom = smf.ols(formula='lfare ~ concen + ldist + ldistsq + y98 + y99 + y00',
              data=airf)
por = pom.fit(cov_type='HC3')
pot = pd.DataFrame({'b'   : round(por.params,5),
                    'se'  : round(por.bse, 5),
                    't'   : round(por.tvalues, 2),
                    'p'   : round(por.pvalues,3)})
print(pot)


# In[6]:


#
# test for period effects
yhyp = ['y98=0','y99=0','y00=0']
ftest = por.f_test(yhyp)
print('Testing year effect in POLS')
print('F-stat : {}'.format(ftest.statistic[0][0]))
print('p-value: {}'.format(ftest.pvalue))


# In[7]:


#
# cluster robust standard errors
#
pom = plm.PooledOLS.from_formula(formula='lfare ~ 1 + concen + ldist + ldistsq + y98 + y99 + y00',
                                data=airf)
por = pom.fit(cov_type='robust')
print(por)
por = pom.fit(cov_type='clustered', cluster_entity=True)
print(por)

yhyp = ['y98=0','y99=0','y00=0']
wtest = por.wald_test(formula=yhyp)
print(wtest)
print('Testing year effect in POLS')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))


# In[9]:


#
# preliminary test for unobserved effects
#
airf['rhat'] = por.resids
airf['lrhat'] = airf.rhat.shift()
pmd = plm.PooledOLS.from_formula(formula='lfare ~ 1 + concen + ldist + ldistsq + y99 + y00 + lrhat',
                                 data=airf[airf['t']>1997])
pmc = pmd.fit(cov_type='clustered', cluster_entity=True)
print(pmc)

uhyp = ['lrhat=0']
wtest = pmc.wald_test(formula=uhyp)
print(wtest)
print('Testing unobserved effects')
print('Chi2   : {}'.format(wtest.stat))
print('p-value: {}'.format(wtest.pval))


# In[10]:


#
# first difference estimator
#
fdm = plm.FirstDifferenceOLS.from_formula(formula='lfare ~ concen',
                                data=airf)
fdr = fdm.fit(cov_type='clustered', cluster_entity=True)
print(fdr)

fdm = plm.FirstDifferenceOLS.from_formula(formula='lfare ~ concen + y98 + y99 + y00',
                                          data=airf)
fdr = fdm.fit(cov_type='clustered', cluster_entity=True)
print(fdr)


# In[11]:


#
# fixed effects estimator
#
fem = plm.PanelOLS.from_formula(
            formula='lfare ~ 1 + concen + EntityEffects + C(t)',
            data=airf)
fer = fem.fit(cov_type='clustered', cluster_entity=True)
print(fer)


# In[12]:


#
# random effects estimator
#
rem = plm.RandomEffects.from_formula(
            formula='lfare ~ 1 + concen + ldist + ldistsq + C(t) + EntityEffects',
            data=airf)
rer = rem.fit(cov_type='clustered', cluster_entity=True)
print(rer)


# In[13]:


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
airf = airf.set_index(['id','year'])


# In[14]:


#
# correlated random effects estimator
#
crm = plm.RandomEffects.from_formula(
            formula='lfare ~ 1 + concen + concen_b + ldist + ldistsq + C(t) + EntityEffects',
            data=airf)
crr = crm.fit(cov_type='clustered', cluster_entity=True)
print(crr)


# In[15]:


#
# comparing results
print(plm.panel.compare({'POLS': por,
                         'FD'  : fdr,
                         'FE'  : fer,
                         'RE'  : rer,
                         'CRE' : crr},
                        precision='std_errors'))


# In[ ]:




