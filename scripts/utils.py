import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_group_contribution(data, grouping_col):
    """Calculates and prints the percentage contribution of each group in a column."""
    group_counts = data[grouping_col].value_counts()
    total_count = len(data)
    for group, count in group_counts.items():
        percentage = (count / total_count) * 100
        print(f"{group}: {percentage:.2f}% of the data")

def create_bar_chart(data, x_col, y_col, title, xlabel, ylabel):
    """Creates and displays a simple bar chart."""
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_col], data[y_col], color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def create_grouped_bar_chart(data, x_col, y_col, group_col, title, xlabel, ylabel):
    """Creates and displays a grouped bar chart."""
    pivot_table = data.pivot_table(index=x_col, columns=group_col, values=y_col, fill_value=0)
    pivot_table.plot(kind='bar', figsize=(12, 7))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def create_boxplot(data, x_col, y_col, title, xlabel, ylabel):
    """ Creates and displays a boxplot chart"""
    plt.figure(figsize=(10,6))
    sns.boxplot(x=x_col, y=y_col, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()