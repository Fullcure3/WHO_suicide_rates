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
desired_columns = ['ParentLocation', 'Location', 'Value', 'Dim1', 'Dim2ValueCode']

suicides_clean = suicides[desired_columns].rename(columns={'Dim1': 'sex', 'Dim2ValueCode': 'age_range', 'ParentLocation': 'region', 'Location': 'country', 'Value': 'suicide_rate'})
suicides_clean.head()

# Adjust sex column where Both sexes = Both for simplicity
suicides_clean['sex'] = suicides_clean.sex.apply(lambda sex: 'Both' if sex == 'Both sexes' else sex)

# Check for correct changes
suicides_clean.sex.value_counts()

# Remove the redundant Both category in sexes for future EDA (Compare Male to Female only)
suicides_clean = suicides_clean[suicides_clean.sex != 'Both']
suicides_clean.head()

suicides_clean.dtypes
suicides_clean.isnull().sum()
suicides_clean.describe()


def summary_stats_barplot(dataframe, np_function, category, value):
    """Takes the dataframe and groups and sorts the data, according to the aggregate function passed in, for seaborn visualization\n"""
    
    # Summary stat to serve as a reference vertical line in the bar graph 
    suicide_rates = np_function(dataframe[value])
    
    # Sort values by category in desending order to prep for creating an ordered barplot
    suicides_sorted = (
                        dataframe.groupby(by=category, as_index=False)[value]
                        .agg(np_function)
                        .sort_values(by=value, ascending=False)
    )
    
    # Visualization of the choosen summary stat to identify trends 
    sns.barplot(data=suicides_sorted, x=value, y=category)
    plt.axvline(suicide_rates, linestyle='--', color='black')
    plt.show()  

# Visualization of the means to identify trends 
summary_stats_barplot(suicides_clean, np.mean, value='suicide_rate', category='region')

# Visualization of the medians to identify trends
summary_stats_barplot(suicides_clean, np.median, value='suicide_rate', category='region')

#Histogram to visualize the spread of the suicide rates of all countries together
sns.histplot(data=suicides_clean, x='suicide_rate')
plt.clf()

#Figure level plot to aid visualization
sns.displot(data=suicides_clean, x='suicide_rate', col='region', col_wrap=3)
plt.close()

def sorted_boxplot(dataframe, category, value):
    """Sorts the dataframe based on the category/column in descending order to create a sorted boxplot visualization"""
    dataframe.sort_values(by=category, ascending=False, inplace=True)
    sns.boxplot(data=dataframe, x=value, y=category)
    plt.show()

sorted_boxplot(suicides_clean, 'region', 'suicide_rate')

sorted_boxplot(suicides_clean, 'age_range', 'suicide_rate')

sorted_boxplot(suicides_clean, 'sex', 'suicide_rate')

# Histogram to view overlap of suicide rates of males vs females
sns.histplot(data=suicides_clean, x='suicide_rate', hue='sex', bins=30)

# Checking for sample sizes between regions. Ideally sample sizes between categories should be close
suicides_clean.region.value_counts()

def unique_categories(dataframe, column):
    """Return a list of all unique categories for a given dataframe column"""
    categories = list(dataframe[column].unique())
    return categories

def column_std(dataframe, column, value):
    """Calculates the std of each category in a column to check for ANOVA Test assumption:\n
    The standard deviations of the groups should be equal\n
    Returns a list of unique categories for the column"""
    
    #List of unique categories in a column for std calculations
    categories = unique_categories(dataframe, column)

    #Prints out category and std for ANOVA Test std assumption 
    for category in categories:
        print(category, dataframe[dataframe[column] == category][value].std())


column_std(suicides_clean, 'region', 'suicide_rate')

### Continue refactor from here

#Use the zscore of the value column to reduce the effects of outliers on assumptions 2 and 3
zscore_standard_threshold = 3
suicides_zscored = suicides_clean[(np.abs(zscore(suicides_clean.suicide_rate)) < zscore_standard_threshold)]
suicides_zscored.region.value_counts()


#Checking std difference after zscore to meet ANOVA assumption 2
column_std(suicides_zscored, 'region', 'suicide_rate')

def anova_test(dataframe, column, value):
    """Preps data then performs an ANOVA Test based on the unique categories for the column of interest\n
    value = numeric data to test the choosen column categories against\n"""
    
    #List of unique categories from a column to prep for data filtering  
    categories = unique_categories(dataframe, column)

    #Data of each unique category to unpack as arguements for f_oneway (One way ANOVA Test)
    category_data = tuple([dataframe[value][dataframe[column] == category] for category in categories])
    
    #ANOVA Test to determine p-value significance
    fstat, pval = f_oneway(*category_data)
    print(pval)

anova_test(suicides_zscored, 'region', 'suicide_rate')

def tukeys_test(dataframe, column, value):
    """Prints out the results of Tukey's Range Test to determine which pairings of an ANOVA Test are significant\n
    Uses standard significance threshold of 0.05"""
       
    pval_threshold = 0.05
    tukey_results = pairwise_tukeyhsd(dataframe[value], dataframe[column], pval_threshold)
    print(tukey_results)

tukeys_test(suicides_zscored, 'region', 'suicide_rate')


# Copy to perform ln transformation to preserve clean dataset
suicides_ln = suicides_clean.copy()

#Remove all suicide rates <=0 to prep for transformation
suicides_ln = suicides_ln[suicides_ln.suicide_rate > 0]

# Check number of records removed
records_removed = len(suicides_clean) - len(suicides_ln)
print(f'{records_removed} records removed')

# ln transformation for data profiling and ANOVA test
suicides_ln['suicide_rate'] = np.log(suicides_ln['suicide_rate'])
suicides_ln.head()

#Calculating std of each region to check if assumption 2 is met (std of groups should be equal)
column_std(suicides_ln, 'region', 'suicide_rate')

#Visualization of ln transformation distribution for assumption 3
sns.boxplot(data=suicides_ln, x='suicide_rate', y='region')

sns.displot(data=suicides_ln, x='suicide_rate', col='region', col_wrap=3)

#Calculating std of each age range to check if assumption 2 is met (std of groups should be equal)
column_std(suicides_ln, 'age_range', 'suicide_rate')

#Visualization of ln transformation distribution for assumption 3
sorted_boxplot(suicides_ln, 'age_range', 'suicide_rate')

#Histogram of each category to visualize each distribution (assumption 3) 
sns.displot(data=suicides_ln, x='suicide_rate', col='age_range', col_wrap=3)

#Calculating std of each sex to check if assumption 2 is met (std of groups should be equal)
column_std(suicides_ln, 'sex', 'suicide_rate')

#Visualization of ln transformation distribution for assumption 3
sorted_boxplot(suicides_ln, 'sex', 'suicide_rate')

# Histogram to view overlap of suicide rates of males vs females
sns.histplot(data=suicides_ln, x='suicide_rate', hue='sex')

# Anova test to determine if the pval is significant
anova_test(suicides_ln, 'age_range', 'suicide_rate')

#Tukey's Range Test to determine which pairings are significant
tukeys_test(suicides_ln, 'age_range', 'suicide_rate')

#Filtering suicide rates by age range to prep for anova test
sexes = unique_categories(suicides_ln, 'sex')
suicide_sexes = {sex:suicides_ln.suicide_rate[suicides_ln.sex == sex] for sex in sexes}
print(suicide_sexes.keys())

#2 Sample T test to determine if the pval is significant
tstat, pval = ttest_ind(suicide_sexes['Male'], suicide_sexes['Female'])
print(pval)