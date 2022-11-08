import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def missing_data_check(dataframe):
    """Summary of dtypes and null values to aid in data cleaning"""
    print(dataframe.info())
    print(dataframe.isnull().sum())


def summary_stats_barplot(dataframe, np_function, category, value):
    """Takes the dataframe and groups and sorts the data, according to the aggregate function passed in, for seaborn visualization\n
    Examples of functions are np.mean, np.median, etc"""
    
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


def sorted_boxplot(dataframe, category, value):
    """Sorts the dataframe based on the category/column in descending order to create a sorted boxplot visualization"""
    dataframe.sort_values(by=category, ascending=False, inplace=True)
    sns.boxplot(data=dataframe, x=value, y=category)
    plt.show()


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


def zscore_normalization(dataframe, column, zscore_threshold=3):
    """Requires from scipy.stats import zscore\n
    Removes all rows that are considered outliers from a specific column based on a zscore threshold.\n
    The values above or below the threshold are considered outliers (default +/- 3)\n
    Prints out the number of rows removed.\n
    Returns the dataframe with outliers removed"""
    
    zscore_threshold
    # Abs to facilitate filtering of values that are above the zscore threshold
    dataframe_zscored = dataframe[(zscore(dataframe[column].abs()) < zscore_threshold)]

    records_removed = len(dataframe) - len(dataframe_zscored)
    print(f'{records_removed} rows removed')
    return dataframe_zscored


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


def two_tail_ttest(dataframe, column, value, **kwargs):
    """Preps data then performs an two_tail TTest based on the binary categories for the column of interest\n
    value = numeric data to test the choosen column categories against\n"""
    
    #List of unique categories from a column to prep for data filtering  
    categories = unique_categories(dataframe, column)

    #Data of each unique category to unpack as arguements for ttest_ind (two tail ttest)
    category_data = tuple([dataframe[value][dataframe[column] == category] for category in categories])
    
    #Two Tail TTest to determine p-value significance
    fstat, pval = ttest_ind(*category_data, **kwargs)
    print(pval)


def tukeys_test(dataframe, column, value, pval_threshold=0.05):
    """Prints out the results of Tukey's Range Test to determine which pairings of an ANOVA Test are significant\n
    Uses standard significance threshold of 0.05 by default"""
       
    pval_threshold
    tukey_results = pairwise_tukeyhsd(dataframe[value], dataframe[column], pval_threshold)
    print(tukey_results)


def ln_transformation(dataframe, column):
    """Performs a natural log transformation on the values from a specific column\n
    Prints out the number of row removed to complete the transformation\n
    Returns a dataframe tranformed column"""

    # Copy to perform ln transformation to preserve clean dataset
    dataframe_ln = dataframe.copy()

    #Remove all suicide rates <=0 to prep for transformation (log of value <= 0 is undefined)
    undefined = 0
    dataframe_ln = dataframe_ln[dataframe_ln[column] > undefined]

    # Check number of records removed
    records_removed = len(dataframe) - len(dataframe_ln)
    print(f'{records_removed} records removed')

    # ln transformation for data profiling and ANOVA test
    dataframe_ln[column] = np.log(dataframe_ln[column])

    return dataframe_ln


