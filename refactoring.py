import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

suicides = pd.read_csv('suicides.csv')
suicides.head()
# Check columns to identify which columns to drop that contain missing data over 60%
suicides.info()

# Drop missing data columns to clean the dataset
suicides.dropna(axis='columns', how='all', inplace=True)
suicides.head()

#Check values of Dim2ValueCode to determine how to split the strings
suicides.Dim2ValueCode.value_counts()

#Replace YEARS with "" to prep for a column split
suicides['Dim2ValueCode'] = suicides['Dim2ValueCode'].replace('YEARS', '', regex=True)
print(suicides['Dim2ValueCode'].value_counts())

#Check dataframe for accurate split and new columns
suicides.head()

# Identify columns to remove to create a clean dataset for EDA
suicide_columns = suicides.columns.to_list()
print(suicide_columns)

#Add desired columns to a list
desired_columns = ['ParentLocation', 'Location', 'Period', 'Value', 'Dim1', 'Dim2ValueCode']

suicides_clean = suicides[desired_columns].rename(columns={'Dim1': 'Sex', 'Dim2ValueCode': 'AgeRange'})
suicides_clean.head()

# Adjust Sex column where Both sexes = Both for simplicity
suicides_clean['Sex'] = suicides_clean.Sex.apply(lambda sex: 'Both' if sex == 'Both sexes' else sex)

# Check for correct changes
suicides_clean.Sex.value_counts()

# Remove the redundant Both category in sexes for future EDA (Compare Male to Female only)
suicides_clean = suicides_clean[suicides_clean.Sex != 'Both']
suicides_clean.head()

suicides_clean.dtypes
suicides_clean.isnull().sum()
suicides_clean.describe()

# Mean suicides rates including all regions to serve as a reference point in the bar graph 
suicide_rates_mean = suicides_clean.Value.mean()

# Sort values for each region in desending order for ordered barplot
suicides_means_sorted = suicides_clean.groupby(by='ParentLocation', as_index=False).Value.mean().sort_values(by='Value', ascending=False)

# Visualization of the means to identify trends
sns.barplot(data=suicides_means_sorted, y='ParentLocation', x='Value')
plt.axvline(suicide_rates_mean, linestyle='--', color='black')

# Median suicides rates including all regions to serve as a reference point in the bar graph 
suicide_rates_median = suicides_clean.Value.median()

# Sort values for each region in desending order for ordered barplot
suicides_median_sorted = suicides_clean.groupby(by='ParentLocation', as_index=False).Value.median().sort_values(by='Value', ascending=False)

# Visualization of the medians to identify trends
sns.barplot(data=suicides_median_sorted, y='ParentLocation', x='Value')
plt.axvline(suicide_rates_mean, linestyle='--', color='black')

#Histogram to visualize the spread of the suicide rates of all countries together
sns.histplot(data=suicides_clean, x='Value')

#Figure level plot to aid visualization
sns.displot(data=suicides_clean, x='Value', col='ParentLocation', col_wrap=3)

suicides_clean.sort_values(by='Value', ascending=False, inplace=True)
sns.boxplot(data=suicides_clean, x='Value', y='ParentLocation')

suicides_clean.sort_values(by='AgeRange', ascending=False, inplace=True)
sns.boxplot(data=suicides_clean, x='Value', y='AgeRange')

suicides_clean.sort_values(by='Sex', ascending=False, inplace=True)
sns.boxplot(data=suicides_clean, x='Value', y='Sex')

# Histogram to view overlap of suicide rates of males vs females
sns.histplot(data=suicides_clean, x='Value', hue='Sex', bins=30)

# Checking for sample sizes between regions. Ideally sample sizes between categories should be close
suicides_clean.ParentLocation.value_counts()

#Calculating std of each region to check if assumption 2 is met (std of groups should be equal)
regions = list(suicides_clean.ParentLocation.unique())

for region in regions:
    print(region, suicides_clean[suicides_clean.ParentLocation == region].Value.std())

#Use the zscore of the value column to reduce the effects of outliers on assumptions 2 and 3
zscore_standard_threshold = 3
suicides_zscored = suicides_clean[(np.abs(zscore(suicides_clean.Value)) < zscore_standard_threshold)]
suicides_zscored.ParentLocation.value_counts()

#Checking std difference after zscore to meet ANOVA assumption 2
for region in regions:
    print(region, suicides_zscored[suicides_zscored.ParentLocation == region].Value.std())

#Checking std difference after zscore to meet ANOVA assumption 2
for region in regions:
    print(region, suicides_zscored[suicides_zscored.ParentLocation == region].Value.std())

#Dividing suicide rates by region to prep for anova test
suicide_regions = {region:suicides_zscored.Value[suicides_zscored.ParentLocation == region] for region in regions}
print(suicide_regions.keys())

#Anova test to determine if the pval is significant
fstat, pval = f_oneway(suicide_regions['Americas'], suicide_regions['Europe'], suicide_regions['Africa'],
                       suicide_regions['South-East Asia'], suicide_regions['Eastern Mediterranean'], suicide_regions['Western Pacific'])
print(pval)

#Tukey's Range Test to determine which pairings are significant
sig_threshold = 0.05
tukey_results = pairwise_tukeyhsd(suicides_zscored.Value, suicides_zscored.ParentLocation, sig_threshold)
print(tukey_results)

# Copy to perform ln transformation to preserve clean dataset
suicides_ln = suicides_clean.copy()

#Remove all suicide rates <=0 to prep for transformation
suicides_ln = suicides_ln[suicides_ln.Value > 0]

# Check number of records removed
records_removed = len(suicides_clean) - len(suicides_ln)
print(f'{records_removed} records removed')

# ln transformation for data profiling and ANOVA test
suicides_ln['Value'] = np.log(suicides_ln['Value'])
suicides_ln.head()

#Calculating std of each region to check if assumption 2 is met (std of groups should be equal)
for region in regions:
    print(region, suicides_ln[suicides_ln.ParentLocation == region].Value.std())

#Visualization of ln transformation distribution for assumption 3
sns.boxplot(data=suicides_ln, x='Value', y='ParentLocation')

sns.displot(data=suicides_ln, x='Value', col='ParentLocation', col_wrap=3)

#Calculating std of each age range to check if assumption 2 is met (std of groups should be equal)
ages = list(suicides_ln.AgeRange.unique())

for age in ages:
    print(age, suicides_ln[suicides_clean.AgeRange == age].Value.std())

#Visualization of ln transformation distribution for assumption 3
suicides_ln.sort_values(by='AgeRange', ascending=False, inplace=True)
sns.boxplot(data=suicides_ln, x='Value', y='AgeRange')

sns.displot(data=suicides_ln, x='Value', col='AgeRange', col_wrap=3)

#Calculating std of each sex to check if assumption 2 is met (std of groups should be equal)
sexes = list(suicides_ln.Sex.unique())

for sex in sexes:
    print(sex, suicides_ln[suicides_clean.Sex == sex].Value.std())

#Visualization of ln transformation distribution for assumption 3
suicides_ln.sort_values(by='Sex', ascending=False, inplace=True)
sns.boxplot(data=suicides_ln, x='Value', y='Sex')

# Histogram to view overlap of suicide rates of males vs females
sns.histplot(data=suicides_ln, x='Value', hue='Sex')

#Filtering suicide rates by age range to prep for anova test
suicide_ages = {age:suicides_ln.Value[suicides_ln.AgeRange == age] for age in ages}
print(suicide_ages.keys())

#Anova test to determine if the pval is significant
fstat, pval = f_oneway(suicide_ages['85PLUS'], suicide_ages['75-84'], suicide_ages['65-74'],
                       suicide_ages['55-64'], suicide_ages['45-54'], suicide_ages['35-44'], 
                       suicide_ages['25-34'], suicide_ages['15-24'])
print(pval)

#Tukey's Range Test to determine which pairings are significant
sig_threshold = 0.05
tukey_results_ln = pairwise_tukeyhsd(suicides_ln.Value, suicides_ln.AgeRange, sig_threshold)
print(tukey_results_ln)

#Filtering suicide rates by age range to prep for anova test
suicide_sexes = {sex:suicides_ln.Value[suicides_ln.Sex == sex] for sex in sexes}
print(suicide_sexes.keys())

#2 Sample T test to determine if the pval is significant
tstat, pval = ttest_ind(suicide_sexes['Male'], suicide_sexes['Female'])
print(pval)