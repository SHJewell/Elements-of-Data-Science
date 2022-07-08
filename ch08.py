import empiricaldist
from empiricaldist import Pmf
from statadict import parse_stata_dict
import gzip
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import tools

dict_file = 'GSS.dct'
data_file = 'GSS.dat.gz'

tools.download('https://github.com/AllenDowney/' +
         'ElementsOfDataScience/raw/master/data/' +
         dict_file)

tools.download('https://github.com/AllenDowney/' +
         'ElementsOfDataScience/raw/master/data/' +
         data_file)

outcomes = [1, 2, 3, 4, 5, 6]
die = Pmf(1/6, outcomes)

stata_dict = parse_stata_dict(dict_file)
fp = gzip.open(data_file)

gss = pd.read_fwf(fp, names=stata_dict.names, colspecs=stata_dict.colspecs)
gss.shape

educ = gss['EDUC'].replace([98, 99], np.nan)

# educ.hist(grid=False)
# plt.xlabel('Years of education')
# plt.ylabel('Number of respondents')
# plt.title('Histogram of education level')
# plt.show()

pmf_educ = Pmf.from_seq(educ, normalize=False)
pmf_educ_norm = Pmf.from_seq(educ, normalize=True)

pmf_educ_norm.bar(label='EDUC')

plt.xlabel('Years of education')
plt.xticks(range(0, 21, 4))
plt.ylabel('PMF')
plt.title('Distribution of years of education')
plt.legend()
plt.show()

print(outcomes)


