import empiricaldist
from empiricaldist import Pmf, Cdf
from statadict import parse_stata_dict
import gzip
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

import tools

def draw_line(p, q, x):
    xs = [q, q, x]
    ys = [0, p, p]
    plt.plot(xs, ys, ':', color='gray')

def draw_arrow_left(p, q, x):
    dx = 3
    dy = 0.025
    xs = [x+dx, x, x+dx]
    ys = [p-dy, p, p+dy]
    plt.plot(xs, ys, ':', color='gray')

def draw_arrow_down(p, q, y):
    dx = 1.25
    dy = 0.045
    xs = [q-dx, q, q+dx]
    ys = [y+dy, y, y+dy]
    plt.plot(xs, ys, ':', color='gray')

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

# pmf_educ_norm.bar(label='EDUC')
# plt.xlabel('Years of education')
# plt.xticks(range(0, 21, 4))
# plt.ylabel('PMF')
# plt.title('Distribution of years of education')
# plt.legend()
# plt.show()

age = gss['AGE'].replace([98, 99], np.nan)

cdf_age = Cdf.from_seq(age)

x = 17
q = 51
p = cdf_age(q)

# cdf_age.plot()
# draw_line(p, q, x)
# draw_arrow_left(p, q, x)
# plt.xlabel('Age (years)')
# plt.xlim(x-1, 91)
# plt.ylabel('CDF')
# plt.title('Distribution of Ages')
# plt.show()

p1 = 0.25
q1 = cdf_age.inverse(p1)

p2 = 0.75
q2 = cdf_age.inverse(p2)

x = 17

# cdf_age.plot()
# draw_line(p1, q1, x)
# draw_arrow_down(p1, q1, 0)
# draw_line(p2, q2, x)
# draw_arrow_down(p2, q2, 0)
# plt.xlabel('Age (years)')
# plt.xlim(x-1, 91)
# plt.ylabel('CDF')
# plt.title('Distribution of Ages')
# plt.show()

income = gss['REALINC'].replace(0, np.nan)
cdf_income = Cdf.from_seq(income)

# cdf_income.plot()
# plt.xlabel('Income ($)')
# plt.ylabel('CDF')
# plt.title('Distribution of Incomes')
# plt.show()

male = (gss['SEX'] == 1)
female = (gss['SEX'] == 2)

male_age = age[male]
female_age = age[female]

pmf_male_age = Pmf.from_seq(male_age)
pmf_female_age = Pmf.from_seq(female_age)

# pmf_male_age.plot(label='Male')
# pmf_female_age.plot(label='Female')
# plt.xlabel('Age (years)')
# plt.ylabel('PMF')
# plt.title('Distribution of age by sex')
# plt.legend()
# plt.show()

cdf_male_age = Cdf.from_seq(male_age)
cdf_female_age = Cdf.from_seq(female_age)

# cdf_male_age.plot(label='Male')
# cdf_female_age.plot(label='Female')
# plt.xlabel('Age (years)')
# plt.ylabel('PMF')
# plt.title('Distribution of age by sex')
# plt.legend()
# plt.show()

pre95 = (gss['YEAR'] < 1995)
post95 = (gss['YEAR'] >= 1995)

# Pmf.from_seq(income[pre95]).plot(label='Before 1995')
# Pmf.from_seq(income[post95]).plot(label='After 1995')
# plt.xlabel('Income (1986 USD)')
# plt.ylabel('PMF')
# plt.title('Distribution of income')
# plt.legend()
# plt.show()

# Cdf.from_seq(income[pre95]).plot(label='Before 1995')
# Cdf.from_seq(income[post95]).plot(label='After 1995')
# plt.xlabel('Income (1986 USD)')
# plt.ylabel('Cdf')
# plt.title('Distribution of income')
# plt.legend()
# plt.show()

high = gss['EDUC'] <= 12
assc = (12 < gss['EDUC']) & (gss['EDUC'] < 16)
bach = gss['EDUC'] >= 16

# Cdf.from_seq(income[high]).plot(label='Highschool Diploma or less')
# Cdf.from_seq(income[assc]).plot(label='Associate')
# Cdf.from_seq(income[bach]).plot(label='Bacehlors')
# plt.xlabel('Income (1986 USD)')
# plt.ylabel('Cdf')
# plt.title('Distribution of by eduction')
# plt.legend()
# plt.show()

np.random.seed(17)
sample = np.random.normal(size=1000)

cdf_sample = Cdf.from_seq(sample)
# cdf_sample.plot(label='Random sample')
# plt.xlabel('x')
# plt.ylabel('CDF')
# plt.legend()
# plt.show()

xs = np.linspace(-3, 3)
ys = norm(0, 1).cdf(xs)

# plt.plot(xs, ys, color='gray', label='Normal CDF')
# cdf_sample.plot(label='Random sample')
# plt.xlabel('x')
# plt.ylabel('CDF')
# plt.legend()
# plt.show()

print(age.describe())

x_age = np.linspace(age.min(), age.max())
y_age = norm(1, age.max()).cdf(x_age)

plt.plot(x_age, y_age, color='gray', label='Normal CDF')
cdf_age.plot(label='US Age')
plt.xlabel('Age (years)')
plt.ylabel('CDF')
plt.legend()
plt.show()

print(outcomes)


