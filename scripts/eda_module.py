import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class EDAAnalyzer:
    """
    A class for performing Exploratory Data Analysis on insurance claim data.
    """

    def __init__(self, df):
        """
        Initializes the EDAAnalyzer with the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing insurance claim data.
        """
        self.df = df.copy()

    def summarize_data(self):
        """
        Performs data summarization: descriptive statistics and data structure review.
        """
        print("Descriptive Statistics:")
        print(self.df.describe())
        print("\nData Types:")
        print(self.df.dtypes)

    def convert_data_types(self):
        """
        Converts columns to the correct types (dates, categories).
        """
        try:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
            self.df['VehicleIntroDate'] = pd.to_datetime(self.df['VehicleIntroDate'], errors='coerce')
        except KeyError as e:
            print(f"Error converting date columns: {e} not found")


        categorical_cols = ['IsVATRegistered', 'Citizenship','LegalType','Title','Language','Bank','AccountType','MaritalStatus',
        'Gender','Country','Province','PostalCode','MainCrestaZone','SubCrestaZone','ItemType','Mmcode','VehicleType','Make',
        'Model','Bodytype','AlarmImmobiliser','TrackingDevice','NewVehicle','WrittenOff','Rebuilt','Converted','CrossBorder',
        'CoverCategory','CoverType','CoverGroup','Section','Product','StatutoryClass','StatutoryRiskType']

        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')

    def assess_data_quality(self):
        """
        Assesses data quality by checking for missing values and handle them.
        """
        print("\nMissing Values per Column:")
        print(self.df.isnull().sum())

        numerical_cols = self.df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if self.df[col].isnull().any():
                median_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_value)


        categorical_cols = self.df.select_dtypes(include='category').columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                mode_value = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_value)

    def univariate_analysis(self):
        
        import math

        numerical_cols = self.df.select_dtypes(include=np.number).columns
        categorical_cols = self.df.select_dtypes(include='category').columns

        # Helper function to plot in batches
        def plot_in_batches(columns, plot_func, title, batch_size=2):
            num_batches = math.ceil(len(columns) / batch_size)
            for batch_idx in range(num_batches):
                batch_cols = columns[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                fig, axes = plt.subplots(1, len(batch_cols), figsize=(12, 5))
                if len(batch_cols) == 1:
                    axes = [axes]  # Make single plot iterable
                for col, ax in zip(batch_cols, axes):
                    plot_func(col, ax)
                plt.suptitle(title, fontsize=14)
                plt.tight_layout()
                plt.show()

        # Numerical columns: Histograms
        def plot_numerical(col, ax):
            sns.histplot(self.df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')

        plot_in_batches(numerical_cols, plot_numerical, 'Numerical Columns')

        # Categorical columns: Bar charts
        def plot_categorical(col, ax):
            self.df[col].value_counts().head(20).plot(kind='bar', ax=ax)
            ax.set_title(f'{col} (Top 20)')
            ax.tick_params(axis='x', rotation=45)

        plot_in_batches(categorical_cols, plot_categorical, 'Categorical Columns')


    def bivariate_analysis(self):
        """
        Performs bivariate analysis: correlation matrix, scatter plots, box plots, and trend analysis.
        """
        # # Proceed with bivariate analysis
        # numerical_cols = self.df.select_dtypes(include=np.number).columns
        # correlation_matrix = self.df[numerical_cols].corr()
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Correlation Matrix')
        # plt.show()

        for cat_col in self.df.select_dtypes(include='category').columns[:5]:
            for num_col in self.df.select_dtypes(include=np.number).columns[:5]:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=cat_col, y=num_col, data=self.df)
                plt.title(f'{num_col} by {cat_col}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

        for cat_col1 in self.df.select_dtypes(include='category').columns[:5]:
            for cat_col2 in self.df.select_dtypes(include='category').columns[5:10]:
                cross_tab = pd.crosstab(self.df[cat_col1], self.df[cat_col2])
                plt.figure(figsize=(10, 6))
                sns.heatmap(cross_tab, cmap='viridis', annot=True)
                plt.title(f'Cross Tabulation of {cat_col1} and {cat_col2}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

        # Trend analysis using 'TransactionMonth'
        if 'TransactionMonth' in self.df.columns:
            monthly_province_data = self.df.groupby(['TransactionMonth', 'Province']).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'PolicyID': 'count'
            }).reset_index()

            plt.figure(figsize=(12, 6))
            sns.lineplot(x='TransactionMonth', y='TotalPremium', hue='Province', data=monthly_province_data)
            plt.title('Total Premium Trends Over Time by Province')
            plt.xlabel('Month')
            plt.ylabel('Total Premium')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 6))
            sns.lineplot(x='TransactionMonth', y='TotalClaims', hue='Province', data=monthly_province_data)
            plt.title('Total Claim Trends Over Time by Province')
            plt.xlabel('Month')
            plt.ylabel('Total Claims')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 6))
            sns.lineplot(x='TransactionMonth', y='PolicyID', hue='Province', data=monthly_province_data)
            plt.title('Number of Policies Trends Over Time by Province')
            plt.xlabel('Month')
            plt.ylabel('Policy Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("Error: 'TransactionMonth' column is missing from the dataset.")


    def outlier_detection(self):
        """
        Detects outliers using box plots.
        """
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            # Drop null values to avoid plotting issues
            column_data = self.df[col].dropna()
            
            if column_data.empty:
                print(f"Skipping column '{col}' as it contains no valid data.")
                continue

            plt.figure(figsize=(8, 5))
            sns.boxplot(x=column_data)
            plt.title(f'Box Plot of {col}')
            plt.show()


    def creative_visualizations(self):
         """
         Generates 3 creative visualizations
         """
         province_risk = self.df.groupby('Province').agg({'TotalClaims':'sum','TotalPremium':'sum'})
         province_risk['ClaimRatio'] = province_risk['TotalClaims']/province_risk['TotalPremium']

         plt.figure(figsize=(10,6))
         sns.barplot(x = province_risk.index, y = province_risk['ClaimRatio'])
         plt.title("Risk Rating by Province")
         plt.xticks(rotation = 45, ha = 'right')
         plt.tight_layout()
         plt.show()

        # 2. Vehicle Type and claims (Example)
         plt.figure(figsize=(10,6))
         sns.barplot(x = 'VehicleType', y = 'TotalClaims', data = self.df)
         plt.title("Claims by Vehicle Type")
         plt.xticks(rotation = 45, ha = 'right')
         plt.tight_layout()
         plt.show()

        # 3. Premium vs Claims Scatter plot (example)
         plt.figure(figsize=(10,6))
         sns.scatterplot(x = 'TotalPremium', y = 'TotalClaims', data = self.df)
         plt.title("Premium vs Claims Scatter Plot")
         plt.tight_layout()
         plt.show()

    def perform_eda(self):
        """
        Performs all EDA steps sequentially.
        """
        self.summarize_data()
        self.convert_data_types()
        self.assess_data_quality()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.outlier_detection()
        self.creative_visualizations()

    def get_processed_df(self):
        """
        Returns the processed DataFrame.
        
        Returns:
             pd.DataFrame: The processed DataFrame
        """
        return self.df