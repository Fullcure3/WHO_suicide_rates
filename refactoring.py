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


def summary_stats_barplot(dataframe, np_function, category, value):
    """Takes the dataframe and groups and sorts the data, according to the aggregate function passed in, for seaborn visualization"""
    
    # Summary stat to serve as a reference vertical line in the bar graph 
    suicide_rates = np_function(dataframe[value])
    
    # Sort values by category in desending order to prep for creating an ordered barplot
    suicides_sorted = (
                        dataframe.groupby(by=category, as_index=False)[value]
                        .agg(np_function)
                        .sort_values(by='Value', ascending=False)
    )
    
    # Visualization of the choosen summary stat to identify trends 
    sns.barplot(data=suicides_sorted, x=value, y=category)
    plt.axvline(suicide_rates, linestyle='--', color='black')
    plt.show()  

# Visualization of the means to identify trends 
summary_stats_barplot(suicides_clean, np.mean, value='Value', category='ParentLocation')

# Visualization of the medians to identify trends
summary_stats_barplot(suicides_clean, np.median, value='Value', category='ParentLocation')

#Histogram to visualize the spread of the suicide rates of all countries together
sns.histplot(data=suicides_clean, x='Value')
plt.clf()

#Figure level plot to aid visualization
sns.displot(data=suicides_clean, x='Value', col='ParentLocation', col_wrap=3)
plt.close()

def sorted_boxplot(dataframe, category, value):
    """Sorts the dataframe based on the category/column in descending order to create a sorted boxplot visualization"""
    dataframe.sort_values(by=category, ascending=False, inplace=True)
    sns.boxplot(data=dataframe, x=value, y=category)
    plt.show()

sorted_boxplot(suicides_clean, 'ParentLocation', 'Value')

sorted_boxplot(suicides_clean, 'AgeRange', 'Value')

sorted_boxplot(suicides_clean, 'Sex', 'Value')

# Histogram to view overlap of suicide rates of males vs females
sns.histplot(data=suicides_clean, x='Value', hue='Sex', bins=30)

# Checking for sample sizes between regions. Ideally sample sizes between categories should be close
suicides_clean.ParentLocation.value_counts()

### Continue refactor from here
def column_std(dataframe, column, num_value):
    """Calculates the std of each category in a column to check for ANOVA Test assumption:\n
    The standard deviations of the groups should be equal\n
    Returns a list of unique categories for the column"""
    
    #List of unique categories in a column for std calculations
    categories = list(dataframe[column].unique())

    #Prints out category and std for ANOVA Test std assumption 
    for category in categories:
        print(category, dataframe[dataframe[column] == category][num_value].std())
    
    return categories


column_std(suicides_clean, 'ParentLocation', 'Value')

#Use the zscore of the value column to reduce the effects of outliers on assumptions 2 and 3
zscore_standard_threshold = 3
suicides_zscored = suicides_clean[(np.abs(zscore(suicides_clean.Value)) < zscore_standard_threshold)]
suicides_zscored.ParentLocation.value_counts()


#Checking std difference after zscore to meet ANOVA assumption 2
regions = column_std(suicides_zscored, 'ParentLocation', 'Value')

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
regions = column_std(suicides_ln, 'ParentLocation', 'Value')

#Visualization of ln transformation distribution for assumption 3
sns.boxplot(data=suicides_ln, x='Value', y='ParentLocation')

sns.displot(data=suicides_ln, x='Value', col='ParentLocation', col_wrap=3)

#Calculating std of each age range to check if assumption 2 is met (std of groups should be equal)
ages = column_std(suicides_ln, 'AgeRange', 'Value')

#Visualization of ln transformation distribution for assumption 3
sorted_boxplot(suicides_ln, 'AgeRange', 'Value')

#Histogram of each category to visualize each distribution (assumption 3) 
sns.displot(data=suicides_ln, x='Value', col='AgeRange', col_wrap=3)

#Calculating std of each sex to check if assumption 2 is met (std of groups should be equal)
sexes = column_std(suicides_ln, 'Sex', 'Value')

#Visualization of ln transformation distribution for assumption 3
sorted_boxplot(suicides_ln, 'Sex', 'Value')

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