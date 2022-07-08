import pandas as pd
import statadict
from statadict import parse_stata_dict
import numpy as np
from matplotlib import pyplot as plt

path = 'E:\\Documents\\Datasets\\Element of Data Science\\2015_2017_FemPregData.dat'
dict_path = 'E:\\Documents\\Datasets\\Element of Data Science\\2015_2017_FemPregSetup.dct'

stata_dict = parse_stata_dict(dict_path)
#nsfg = pd.read_fwf(path, colspec=statadict.colspecs)
nsfg = pd.read_fwf(path,
                   names=stata_dict.names,
                   colspecs=stata_dict.colspecs)

pounds = nsfg['BIRTHWGT_LB1']
pounds_clean = nsfg['BIRTHWGT_OZ1'].replace([98, 99], np.nan)

print(pounds.value_counts())
print(pounds.value_counts().sort_index())
print(pounds.describe())
print(pounds_clean.describe())

kilos = pounds_clean / 2.2
ounces = nsfg['BIRTHWGT_OZ1']
ounces_clean = nsfg['BIRTHWGT_OZ1'].replace([98, 99], np.nan)
birth_weight = pounds_clean + ounces_clean / 16

birth_weight.hist(bins=30)
plt.xlabel('Birth weight in pounds')
plt.ylabel('Number of live births')
plt.title('Distribution of U.S. birth weight')
#plt.show()

nsfg['AGECON'].hist(bins=20)
plt.xlabel('Mother\'s Age at Conception')
plt.ylabel('Number of Births')
#plt.show()

preterm = (nsfg['PRGLNGTH'] < 37)
preterm.dtype

live = (nsfg['OUTCOME'] == 1)

live_preterm = (live & preterm)
live_preterm.mean()

print((preterm.value_counts()[True] / live.value_counts()[True]) * 100)
print(preterm)

preterm_weight = birth_weight[preterm]
preterm_weight.mean()

fullterm = (nsfg['PRGLNGTH'] >= 37)
fullterm_weight = birth_weight[fullterm]
fullterm_weight.mean()

multis = live[nsfg['NBRNALIV'] > 1]
print(multis.value_counts()[True] / live.value_counts()[True])

single = live[nsfg['NBRNALIV'] == 1]
print(preterm.value_counts()[True] / single.value_counts()[True])

#weighted mean
sampling_weight = nsfg['WGT2015_2017']
missing = birth_weight.isna()

valid = birth_weight.notna()
selected = valid & live & single & fullterm

weight_avg = birth_weight * sampling_weight * valid
selected_weights = sampling_weight * valid

print(nsfg)
