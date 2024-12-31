import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import f_oneway, ttest_ind

def perform_t_test(group_a, group_b):
    """Performs an independent samples t-test."""
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    return t_stat, p_value

def perform_z_test(group_a, group_b):
    """ Performs a z-test """
    z_stat, p_value = sm.stats.ztest(group_a, group_b, alternative='two-sided')
    return z_stat, p_value

def perform_anova_test(dependent_col: str, independent_col: str, data: pd.DataFrame):
    """
    Performs a one-way ANOVA test to determine if there are statistically significant differences 
    between the means of the dependent variable across the groups formed by the independent variable.
    
    Parameters:
    ----------
    dependent_col : str
        The name of the column containing the dependent variable (numeric) to test.
    independent_col : str
        The name of the column containing the independent variable (categorical) used to group the data.
    data : pd.DataFrame
        The DataFrame containing the dataset with both the dependent and independent variables.
    
    Returns:
    -------
    f_statistics : float
        The F-statistic value from the ANOVA test.
    p_value : float
        The p-value from the ANOVA test, which indicates the statistical significance.
    """

    # get the grouping along the independet_col
    data_groups = data.groupby(by=independent_col)

    # get the group names formed by using the independet_col
    group_names = list(data_groups.groups.keys())
    
    # get the groups and then put the dependent col values into a global list
    data_points = []
    for group_name in group_names:
        # group dataframe
        group_data = data_groups.get_group(name=group_name)

        # add the data points of the dependent column
        values = group_data[dependent_col]
        data_points.append(values)
    
    # perform an ANOVA test
    f_statistics, p_value = f_oneway(*data_points)

    return f_statistics, p_value
    return f_statistic, p_value
def perform_anova_test_regularized(data, dependent_col, independent_col):
    """Performs an ANOVA test with regularization"""
    model = smf.ols(f'{dependent_col} ~ C({independent_col})', data=data)

    # Fit the model with L1 regularization using fit_regularized
    results = model.fit_regularized(alpha=0.01, L1_wt=1) #Adjust alpha and L1_wt

    # Anova table
    anova_table = sm.stats.anova_lm(results, typ=2)
    f_statistic = anova_table['F'][0]
    p_value = anova_table['PR(>F)'][0]
    return f_statistic, p_value
def perform_kruskal_wallis(data, dependent_col, independent_col):
    """Performs the Kruskal-Wallis H-test for non-parametric data."""
    groups = data[independent_col].unique()
    samples = [data[data[independent_col] == g][dependent_col] for g in groups]
    h_stat, p_value = stats.kruskal(*samples)
    return h_stat, p_value


def test_hypothesis(null_hypothesis, p_value, threshold=0.05):
  """Accepts or rejects a null hypothesis based on the p-value."""
  if p_value < threshold:
    print(f"Rejected the null hypothesis: {null_hypothesis}")
  else:
    print(f"Accepted the null hypothesis: {null_hypothesis}")
  print(f"p_value: {p_value}\n")


def ab_test(data, dependent_col, independent_col, threshold=0.05):
    """Performs an A/B test and outputs the results."""
    unique_values = data[independent_col].unique()
    if len(unique_values) != 2:
        raise ValueError("Independent column must have 2 values for A/B test")
    try:
      group_a = data[data[independent_col] == unique_values[0]][dependent_col]
      group_b = data[data[independent_col] == unique_values[1]][dependent_col]
    except Exception as e:
       print(f"An error has occured when grouping the data:{e}")
       return None, None
    t_stat, p_value = perform_t_test(group_a, group_b)
    return t_stat, p_value