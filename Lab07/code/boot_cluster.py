# ---------------------------------------------------------
# Example of bootstrapping
# Efron's resampling with replacement
#   panel data, w/cluster sampling
#
# Olvar Bergland ECN301 March 2021
#

#
import numpy as np
import pandas as pd
import sys
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#
# load data
rice = pd.read_csv('./rice2.csv')
print(rice.info())
#
# create new variables
rice['lnQ'] = np.log(rice['prod'])
rice['lnD'] = np.log(rice['area'])
rice['lnL'] = np.log(rice['labor'])
rice['lnF'] = np.log(rice['fert'])


#
# CD model specification
cdmod = smf.ols(formula= 'lnQ ~ lnL + lnD + lnF',data=rice)

print()
print('Parameter estimates for lnF')
print()


#
# compare OLS and bootstrap results
cdres = cdmod.fit()
print('OLS estimate and regular standard errors')
print('    parameter: %6.4f' % (cdres.params[3]))
print('    std error: %6.4f' % (cdres.bse[3]))
print()

#
# robust standard errors
cdrob = cdmod.fit(cov_type='HC3')
print('OLS estimate, robust standard errors')
print('    parameter: %6.4f' % (cdrob.params[3]))
print('    std error: %6.4f' % (cdrob.bse[3]))
print()

#
# get cluster robust std errors
cdcrb = cdmod.fit(cov_type='cluster', cov_kwds={'groups': rice['farmid']})
print('OLS estimate and cluster robust standard errors')
print('    parameter: %6.4f' % (cdcrb.params[3]))
print('    std error: %6.4f' % (cdcrb.bse[3]))
print()

#
# bootstrap scale estimate, but recognize the panel data structure
#

#
# create dataframe to keep results
boot_df  = pd.DataFrame(columns=['rep','lnL','lnD','lnF'])


#
# use unique farmids
farmers = rice[['farmid']].drop_duplicates()

b_reps = 10000
for i in range(b_reps):
    bfarms = farmers.sample(frac=1,replace=True)
    bdf = None
    for f, fid in bfarms.iterrows():
        if bdf is None:
            bdf = rice[rice['farmid']==fid['farmid']]
        else:
            bdf = bdf.append(rice[rice['farmid']==fid['farmid']])

    br = smf.ols(formula= 'lnQ ~ lnL + lnD + lnF',data=bdf).fit()
    boot_df  = boot_df.append({'rep': i,'lnL': br.params[1], 'lnD': br.params[2],'lnF': br.params[3]},ignore_index=True)


print('Bootstrap estimate (reps=%3d)' % (b_reps))
print('    parameter: %6.4f' % (boot_df['lnF'].mean()))
print('    std error: %6.4f' % (boot_df['lnF'].std()))
print()


#
# plot distribution of parameter estimates
fig, ax = plt.subplots(figsize=(9,6))

#
# histogram of bootstrap
num_bins = 33
n, bins, patches = ax.hist(boot_df['lnF'], num_bins, density=True)

#
#
prange = np.arange(-0.05, 0.45, 0.005)
#
# OLS estimate, normal distribution
mu  = cdres.params[3]
sig = cdres.bse[3]
y = ((1/(np.sqrt(2*np.pi)*sig)) * np.exp(-0.5*(1/sig*(prange-mu))**2))
ax.plot(prange, y, label='Regular std err')

#
# OLS estimate, normal distribution
#mu  = cdrob.params[3]
#sig = cdrob.bse[3]
#y = ((1/(np.sqrt(2*np.pi)*sig)) * np.exp(-0.5*(1/sig*(bins-mu))**2))
#ax.plot(bins, y, label='HC3 robust std err')

#
# OLS estimate, normal distribution
mu  = cdcrb.params[3]
sig = cdcrb.bse[3]
y = ((1/(np.sqrt(2*np.pi)*sig)) * np.exp(-0.5*(1/sig*(prange-mu))**2))
ax.plot(prange, y, label='Cluster robust std err')

#
ax.set_ylabel('Probability density')
ax.set_xlabel('Regression Parameter')
ax.set_title('Bootstrapped parameter estimate')
fig.tight_layout()
ax.legend()
fig.savefig('rice_c_lnF.png')

