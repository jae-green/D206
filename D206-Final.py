# -*- coding: utf-8 -*-
"""
Final Exam - Code  
D206 - Data Cleaning - Code Addendum
Jessa Green - SID# 71314
Western Governors University
"""

# Import python libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import missingno as msno
import scipy.stats as stats
import seaborn as sns
from fancyimpute import KNN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Read in the Churn Data Set
raw_df = pd.read_csv('churn_raw_data.csv')
pd.options.display.max_columns = None
print(raw_df.head())
print(raw_df.info())

# %%

# Rename columns based on Overview Table "New Names" in final paper
col_renaming = {'CaseOrder': 'case_order',
                'Customer_id': 'cust_id',
                'Interaction': 'interaction_id',
                'City': 'city',
                'State': 'state',
                'County': 'county',
                'Zip': 'zip',
                'Lat': 'latitude',
                'Lng': 'longitude',
                'Population': 'population',
                'Area': 'area',
                'Timezone': 'timezone',
                'Job': 'job',
                'Children': 'children',
                'Age': 'age',
                'Education': 'education',
                'Employment': 'employment',
                'Income': 'income',
                'Marital': 'marital',
                'Gender': 'gender',
                'Churn': 'churn',
                'Outage_sec_perweek': 'outage_sec_wk',
                'Email': 'email_contact_yr',
                'Contacts': 'support_reqs_total',
                'Yearly_equip_failure': 'equip_failure_yr',
                'Techie': 'cust_is_techie',
                'Contract': 'contract_term',
                'Port_modem': 'portable_modem',
                'Tablet': 'tablet',
                'InternetService': 'internet_service',
                'Phone': 'phone_service',
                'Multiple': 'multi_ph_lines',
                'OnlineSecurity': 'online_security',
                'OnlineBackup': 'online_backup',
                'DeviceProtection': 'device_protection',
                'TechSupport': 'tech_support',
                'StreamingTV': 'streaming_tv',
                'StreamingMovies': 'streaming_movies',
                'PaperlessBilling': 'paperless_bill',
                'PaymentMethod': 'pay_method',
                'Tenure': 'tenure',
                'MonthlyCharge': 'monthly_charge',
                'Bandwidth_GB_Year': 'bandwidth_gb_yr',
                'item1': 'surv_timely_resp',
                'item2': 'surv_timely_fix',
                'item3': 'surv_timely_replace',
                'item4': 'surv_reliability',
                'item5': 'surv_options',
                'item6': 'surv_respectful',
                'item7': 'surv_courteous',
                'item8': 'surv_active_listening'}

raw_df.rename(columns=col_renaming, inplace=True)
print(raw_df.info())



# %%
# Detecting and Treating Duplicates


# Define a function to compare all columns and identify full-column duplication
# Code concept from Geeks4Geeks website. See source in Section G

def FindDupCols(df):
    dup_cols = {}
    for columns in range(df.shape[1]):
        col1 = df.iloc[:, columns]
        for other_cols in range(columns + 1, df.shape[1]):
            col2 = df.iloc[:, other_cols]
            if col1.equals(col2):
                col1_name = df.columns.values[columns]
                col2_key = df.columns.values[other_cols]
                dup_cols[col1_name]=(col2_key)
    return dup_cols


# Run the function against our raw dataframe to identify any duplicate columns
dup_cols = FindDupCols(raw_df)
print("Duplicated Columns in Data Set:")
print(dup_cols)

# Drop duplicate column
raw_df = raw_df.drop('Unnamed: 0', 1)
# Confirm the number of columns is reduced from 52 to 51 post-deletion
raw_df.shape

# Check for identical rows across the dataset
duplicate_rows = raw_df.duplicated(keep=False)
raw_df[duplicate_rows]


# Check for duplicate customer IDs
duplicate_rows = raw_df.duplicated(subset='cust_id', keep=False)
raw_df[duplicate_rows]

# Check for rows that are appear identical without matching on customer ID, which could mean one customer was assigned multiple IDs
column_names = ['city', 'state', 'zip', 'job', 'children', 'gender', 'employment', 'income', 'cust_is_techie', 'contract_term', 'monthly_charge']
duplicate_rows = raw_df.duplicated(subset=column_names, keep=False)
raw_df[duplicate_rows]

# %%
# Detecting and Treating Missing Values

# Visualize missing values across data set 
raw_df.isnull().sum()
msno.matrix(raw_df);
msno.heatmap(raw_df);

# View histograms of numeric attributes to visually identify potentially misleading values 
# i.e. chosen default value for missing data
raw_df.hist(figsize=(15,15));

# Check outage_sec_wk to see if the bi-nomial distribution is due to a misleading value 
# i.e. chosen default value for missing data
print(raw_df['outage_sec_wk'].value_counts().head(10))

# Fill in missing categorical values using univariate .mode()[0]
df_imputed = raw_df.copy(deep=True)
df_imputed['cust_is_techie'] = df_imputed['cust_is_techie'].fillna(df_imputed['cust_is_techie'].mode()[0])
df_imputed['phone_service'] = df_imputed['phone_service'].fillna(df_imputed['phone_service'].mode()[0])
df_imputed['tech_support'] = df_imputed['tech_support'].fillna(df_imputed['tech_support'].mode()[0])


# Create df copy for KNN algo imputation
df_knn_imputed = df_imputed.copy(deep=True)

# Encode all categorical values to numeric for KNN imputation process
# Create a loop to encode all categorical columns
oedict = {}
cat_cols = list(df_knn_imputed.select_dtypes(include='object'))

# Loop through categorical columns and encode them with numeric values
for col_name in cat_cols:
    oedict[col_name] = OrdinalEncoder()
    col = df_knn_imputed[col_name]
    col_notnulls = col[col.notnull()]
    reshaped_vals = col_notnulls.values.reshape(-1, 1)
    encoded_vals = oedict[col_name].fit_transform(reshaped_vals)
    df_knn_imputed.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)

# Create visual plot of colums to be imputed
compare_imputes = df_knn_imputed[['children','age','income','tenure','bandwidth_gb_yr']]
compare_imputes.hist(figsize=(15,12));

# Impute missing values with KNN algorithm
knn_imputer = KNN()
df_knn_imputed.iloc[:, :] = np.round(knn_imputer.fit_transform(df_knn_imputed))

# Create visual plot of colums after imputation for comparison
compare_imputes = df_knn_imputed[['children','age','income','tenure','bandwidth_gb_yr']]
compare_imputes.hist(figsize=(15,12));

# Inverse transform the categorical values back to their original values
for col_name in cat_cols:
    reshaped = df_knn_imputed[col_name].values.reshape(-1, 1)
    df_knn_imputed[col_name] = oedict[col_name].inverse_transform(reshaped)


# Update df_imputed dataset with knn updated columns
for col in compare_imputes:
    df_imputed[col] = compare_imputes[col]


# Confirm all nulls removed from dataset
msno.matrix(df_knn_imputed)
df_imputed.isnull().sum()
df_imputed.info()



# %%
# Detecting and Treating Outliers

# Make copy of df for outliers
df_no_outliers = df_imputed.copy(deep=True)

# Use the .describe & .hist methods to visualize the core statistics for each numeric attribute in the data set
df_no_outliers.describe()
df_no_outliers.hist(figsize=(15,15));

# Check for outliers in income - zscore & boxplot
df_no_outliers['zscore_income'] = stats.zscore(df_no_outliers['income'])
income_outliers = df_no_outliers.query('zscore_income > 3 | zscore_income < -3')
income_outliers_count = income_outliers.shape[0]
df_count = df_no_outliers.shape[0]
print('Total outliers in the support_reqs_total column: ', income_outliers_count)
print(income_outliers_count,'is approximately', round(income_outliers_count/df_count,4),'% of the dataset.')
print(income_outliers.shape[0])
plt.hist(df_no_outliers['zscore_income']);

# Exclude the rows of income outliers from primary dataset, placing them in a separate dataset
df_excluded_outliers = df_no_outliers.query('zscore_income > 3 | zscore_income < -3')
df_no_outliers = df_no_outliers[df_no_outliers['zscore_income'] < 3]
df_no_outliers = df_no_outliers[df_no_outliers['zscore_income'] > -3]
df_no_outliers.drop(['zscore_income'], axis=1, inplace=True)
df_no_outliers.info()


# Boxplot for monthly_charge
plt.figure(figsize = (15,3))
boxplot = sns.boxplot(x='monthly_charge', data=df_no_outliers)


# Check for outliers in support_reqs_total - zscore & boxplot
df_no_outliers['zscore_support_reqs'] = stats.zscore(df_no_outliers['support_reqs_total'])
spt_reqs_outliers = df_no_outliers.query('zscore_support_reqs > 3 | zscore_support_reqs < -3')
spt_reqs_count = spt_reqs_outliers.shape[0]
df_count = df_no_outliers.shape[0]
print('Total outliers in the support_reqs_total column: ', spt_reqs_count)
print(spt_reqs_count,'is approximately', round(spt_reqs_count/df_count,4)*100,'% of the dataset.')
plt.figure(figsize = (15,3))
boxplot = sns.boxplot(x='zscore_support_reqs', data=df_no_outliers)
# Exclude the rows of support_reqs_total outliers from primary dataset, placing them in a separate dataset
df_excluded_outliers = df_excluded_outliers.append(spt_reqs_outliers)
df_excluded_outliers.shape[0]
df_no_outliers.drop(df_no_outliers[df_no_outliers['zscore_support_reqs'] > 3].index, inplace=True)
df_no_outliers.drop(df_no_outliers[df_no_outliers['zscore_support_reqs'] < -3].index, inplace=True)
df_no_outliers.drop(['zscore_support_reqs'], axis=1, inplace=True)
df_no_outliers.info()


# Check for outliers in equip_failure_yr - zscore & boxplot
df_no_outliers['zscore_equip_fail'] = stats.zscore(df_no_outliers['equip_failure_yr'])
equip_fail_outliers = df_no_outliers.query('zscore_equip_fail > 3 | zscore_equip_fail < -3')
equip_fail_count = equip_fail_outliers.shape[0]
df_count = df_no_outliers.shape[0]
print('Total outliers in the equip_failure_yr column: ', equip_fail_count)
print(equip_fail_count,'is approximately', round(equip_fail_count/df_count,4)*100,'% of the dataset.')
plt.figure(figsize = (15,3))
boxplot = sns.boxplot(x='zscore_equip_fail', data=df_no_outliers)
# Exclude the rows of equip_failure_yr outliers from primary dataset, placing them in a separate dataset
edf_excluded_outliers = df_excluded_outliers.append(equip_fail_outliers)
df_excluded_outliers.shape[0]
df_no_outliers.drop(df_no_outliers[df_no_outliers['zscore_equip_fail'] > 3].index, inplace=True)
df_no_outliers.drop(df_no_outliers[df_no_outliers['zscore_equip_fail'] < -3].index, inplace=True)
df_no_outliers.drop(['zscore_equip_fail'], axis=1, inplace=True)
df_no_outliers.info()


# Check for outliers in children - zscore & boxplot
df_no_outliers['zscore_children'] = stats.zscore(df_no_outliers['children'])
child_outliers = df_no_outliers.query('zscore_children > 3 | zscore_children < -3')
child_count = child_outliers.shape[0]
df_count = df_no_outliers.shape[0]
print('Total outliers in the children column: ', child_count)
print(child_count,'is approximately', round(child_count/df_count,4)*100,'% of the dataset.')
plt.figure(figsize = (15,3))
boxplot = sns.boxplot(x='zscore_children', data=df_no_outliers)
df_no_outliers.drop(['zscore_children'], axis=1, inplace=True)
# # Exclude the rows of equip_failure_yr outliers from primary dataset, placing them in a separate dataset
# df_excluded_outliers = df_excluded_outliers.append(child_outliers)
# df_excluded_outliers.shape[0]
# df_no_outliers.drop(df_no_outliers[df_no_outliers['zscore_children'] > 3].index, inplace=True)
# df_no_outliers.drop(df_no_outliers[df_no_outliers['zscore_children'] < -3].index, inplace=True)
# df_no_outliers.info()


df_no_outliers.hist(figsize=(15,15));







# # %%
# # Other Data Cleaning Tasks to prepare data set for analysis


# # Create copy of data set to perform remaining cleaning steps on
# clean_df = raw_df.copy(deep=True)

# # children, age, tenure from float to int
# clean_df[['children', 'age', 'tenure']] = clean_df[['children', 'age', 'tenure']].astype(int)

# # Convert zip code to string and fill missing chars with '0'
# clean_df.zip = clean_df.zip.astype(str).str[:-2].str.pad(5,fillchar='0')
# clean_df.zip.sample(20)

# # Replace any invalid zip codes with NaN
# clean_df.zip = clean_df.zip.replace('0000n', np.nan)








# # %%
# # Export cleaned / prepped dataset to CSV for submission
# df.to_csv(r'path/filename.csv')













