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
from scipy.stats import norm, skew
import seaborn as sb
import statsmodels.api as sm
from fancyimpute import KNN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

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

# Display null values across data set 
msno.matrix(raw_df)
raw_df.isnull().sum()

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


# Encode all categorical values to numeric for KNN imputation process
# Create a loop to encode all categorical columns
oedict = {}
cat_cols = list(df_imputed.select_dtypes(include='object'))

# Loop through categorical columns and encode them with numeric values
for col_name in cat_cols:
    oedict[col_name] = OrdinalEncoder()
    col = df_imputed[col_name]
    col_notnulls = col[col.notnull()]
    reshaped_vals = col_notnulls.values.reshape(-1, 1)
    encoded_vals = oedict[col_name].fit_transform(reshaped_vals)
    df_imputed.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)

# Create visual plot of colums to be imputed
compare_imputes = df_imputed[['children','age','income','tenure','bandwidth_gb_yr']]
compare_imputes.hist(figsize=(15,12));

# Impute missing values with KNN algorithm
knn_imputer = KNN()
df_imputed.iloc[:, :] = np.round(knn_imputer.fit_transform(df_imputed))

# Create visual plot of colums after imputation for comparison
compare_imputes = df_imputed[['children','age','income','tenure','bandwidth_gb_yr']]
compare_imputes.hist(figsize=(15,12));

# Inverse transform the categorical values back to their original values
for col_name in cat_cols:
    reshaped = df_imputed[col_name].values.reshape(-1, 1)
    df_imputed[col_name] = oedict[col_name].inverse_transform(reshaped)

# Confirm all nulls removed from dataset
msno.matrix(df_imputed)
df_imputed.isnull().sum()
df_imputed.info()




# %%
# Detecting and Treating Outliers


# # Use the .describe method to visualize the core statistics for each numeric attribute in the data set
# numeric_subset_df.describe()

# # View histograms of numeric attribures to visually identify potential outliers - check for extreme values beyond skew
# numeric_subset_df.hist(figsize=(15,15));

# # Boxplot for income
# plt.figure(figsize = (15,3))
# boxplot = sb.boxplot(x='income', data=raw_df)

# # Boxplot for monthly_charge
# plt.figure(figsize = (15,3))
# boxplot = sb.boxplot(x='monthly_charge', data=raw_df)

# # Boxplot for support_reqs_total
# plt.figure(figsize = (15,3))
# boxplot = sb.boxplot(x='support_reqs_total', data=raw_df)

# # Boxplot for equip_failure_yr
# plt.figure(figsize = (15,3))
# boxplot = sb.boxplot(x='equip_failure_yr', data=raw_df)












# # %%
# # Other Data Cleaning Tasks to prepare data set for analysis


# # Create copy of data set to perform remaining cleaning steps on
# clean_df = raw_df.copy(deep=True)

# # Convert zip code to string and fill missing chars with '0'
# clean_df.zip = clean_df.zip.astype(str).str[:-2].str.pad(5,fillchar='0')
# clean_df.zip.sample(20)

# # Replace any invalid zip codes with NaN
# clean_df.zip = clean_df.zip.replace('0000n', np.nan)

















