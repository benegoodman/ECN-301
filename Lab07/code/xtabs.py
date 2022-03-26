#!/usr/bin/env python
# coding: utf-8
#

import pandas as pd
import wooldridge as woo
mroz = woo.dataWoo('mroz')

#
# basic table of value
print(mroz['inlf'].value_counts())

#
# a one way table
owt = pd.crosstab(mroz['inlf'],columns='count')
print(owt)

#
# a one way table
print(pd.crosstab(mroz['inlf'],columns='count',margins=True))

#
# one way table with frequencies
print(pd.crosstab(mroz['inlf'],columns='count',normalize='all'))

#
# one way table with frequencies
print(pd.crosstab(mroz['educ'],columns='count',normalize='all'))

#
# a two way table
print(pd.crosstab(mroz['educ'],mroz['inlf'],margins=True))

#
# a two way table with frequencies
print(pd.crosstab(mroz['educ'],mroz['inlf'],normalize='all',margins=True))
