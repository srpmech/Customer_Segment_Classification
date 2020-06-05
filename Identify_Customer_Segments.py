#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')

feat_info_function = feat_info


# In[3]:


azdias.shape


# In[4]:


feat_info.shape


# In[5]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

azdias.head(10)


# In[6]:


azdias.head().T


# In[7]:


azdias.describe().T


# In[8]:


feat_info.describe().T


# In[9]:


feat_info.head(85)


# In[10]:


feat_info.info()


# In[11]:


azdias.info()


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[12]:


null_row = azdias.isnull().sum()[azdias.isnull().sum() != 0]

null_row


# In[13]:


null_row.shape


# In[14]:


null_row.index


# In[15]:


# Identify missing or unknown data values and convert them to NaNs.

# Check columns with missing values before any data processing 

null_sum = {'total null': null_row.values,'percentage % of null': np.round(null_row.values/azdias.shape[0]*100,2)}

azdias_null = pd.DataFrame(data=null_sum, index=null_row.index)
azdias_null.sort_values(by='total null', ascending=False, inplace=True)
azdias_null


# In[16]:


azdias_null.shape


# In[17]:


def convert_to_nan(df):
    for i,N in enumerate(df.iteritems()):
        missing = feat_info_function['missing_or_unknown'][i]
        column_name = N[0]
  #      print(missing_unknown)
        missing = missing [1:-1].split(',') # converting all to string for comparison
 #       print(missing_unknown)
        if missing != ['']:
            to_nan = []
            for x in missing:
                if x in ['X','XX']:
                    to_nan.append(x) #adding quotes to x to compare in replace functions
                else:
                    to_nan.append(int(x)) # converting back to int
            

            df[column_name] = df[column_name].replace(to_nan,np.nan) # replace function used for column operation
           
    return df


# In[18]:


azdias_nan = convert_to_nan(azdias)


# In[19]:


azdias_nan.head(50) #Displaying top 50 


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[20]:


null_row_analysis =  azdias_nan.isnull().sum()[azdias_nan.isnull().sum() != 0]

null_row_analysis


# In[21]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

null_analysis = {'total null': null_row_analysis.values,'percentage % of null': np.round(null_row_analysis.values/azdias.shape[0]*100,2)}

azdias_null_analysis = pd.DataFrame(data=null_analysis, index=null_row_analysis.index)
azdias_null_analysis.sort_values(by='total null', ascending=False, inplace=True)
azdias_null_analysis


# In[22]:


plt.hist(azdias_nan[['TITEL_KZ']]);


# In[23]:


plt.hist(azdias_nan[['AGER_TYP']]);


# In[24]:


#Comparing top 9 data with nulls

azdias_nan[['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX','GEBURTSJAHR','ALTER_HH','KKK','REGIOTYP','W_KEIT_KIND_HH']]


# In[25]:


# Investigate patterns in the amount of missing data in each column.

azdias_null_analysis.describe()


# In[26]:


azdias_null_analysis.info()


# Out of 85 features ; 61 features have atleast 1 Nan value in the azdias_nan dataset after initial pre-processing. 
# 
# The first 4 features are above the average number of the null values.

# In[27]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

azdias_condensed = azdias_nan.drop(['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX'],axis = 1)

azdias_condensed


# Removing corresponding features as well.

# In[28]:


feat_info = feat_info[~feat_info['attribute'].isin(['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX'])]


# In[29]:


feat_info.head()


# In[30]:


feat_info.shape


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# Out of 85 features ; 61 features have atleast 1 Nan value in the azdias_nan dataset after initial pre-processing. 
# 
# The first 4 features are above the average number of the null values. This means that out of 85 features 81 can be used for the dataset analysis.
# 
# I am omitting the features that have over 50% values as null to perform my initial analysis. Based on above visualization, seems the same rows are mostly null on these 4 features.
# 

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[31]:


# How much data is missing in each row of the dataset?

nan_row = azdias_condensed.isnull().sum(axis=1)

nan_row_descend = azdias_condensed.isnull().sum(axis=1)

nan_row_descend.sort_values(ascending=False, inplace=True)

nan_row_descend


# In[32]:


nan_row


# In[33]:


nan_row.describe()


# In[34]:


plt.figure(figsize=(16,8))
plt.hist(nan_row, bins=np.arange(0,50,1))
plt.xlabel('nan count')
plt.ylabel('row count')
plt.xticks(np.arange(0,50,5));


# In[35]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.

# function is used to generalize operation

def data_divide (df, missing_values):
    row_count = df.isnull().sum(axis=1)
    c = 0
    for i in row_count:
        if i > missing_values:
            c=c+1
    return c

counter = data_divide(azdias_condensed,25) # Using 25 nan values as the divider for 2 subsets

print('rows with many missing values:',counter)
print('Percentage rows with many missing values:',np.round(counter/azdias.shape[0]*100,2),'%')
    


# In[36]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.



def compare(df,column):
    
    rows_missing = nan_row[nan_row > 25]
    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Many rows missing')
    sns.countplot(df.loc[rows_missing.index,column])
 #   print(df.loc[138,column])
    rows_not_missing = nan_row[nan_row <= 25]
    fig = plt.figure(figsize=(16,4))
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Few rows missing')
    sns.countplot(df.loc[rows_not_missing.index,column])
 #   print(df.loc[rows_not_missing.index,column].shape)

    fig.suptitle(column)
    plt.show()


# In[37]:


compare(azdias_condensed,'LP_STATUS_FEIN')


# In[38]:


compare(azdias_condensed,'ALTERSKATEGORIE_GROB')


# In[39]:


compare(azdias_condensed,'GFK_URLAUBERTYP')


# In[40]:


compare(azdias_condensed,'CJT_GESAMTTYP')


# In[41]:


compare(azdias_condensed,'HH_EINKOMMEN_SCORE')


# In[42]:


rows_missing = nan_row[nan_row > 25]

azdias_optimized = azdias_condensed[~azdias_condensed.index.isin(rows_missing.index)]


# In[43]:


# for later analysis

azdias_missing = azdias.iloc[rows_missing.index]


# In[44]:


#Optimized dataset after dropping multiple rows > 25 nan values

azdias_optimized.shape


# In[45]:


azdias = azdias_optimized #reassigning to original dataset name


# In[46]:


azdias.shape


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# 93318 records or 10.47 % of all rows are missing 25 or more column values as nan. 
# 
# A countplot was created was created for columns with ~ 2% loss of columns values to found the corresponding row values.  
# 
# All 5 columns tested had variation between rows of missing and less missing for 25 nans and so we will keep the rows on the corresponding columns in order to prevent any bias on the clustering analysis

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[47]:


feat_info.shape[0]


# In[48]:


feat_info.head()


# In[49]:


# How many features are there of each data type?

feat_info['type'].value_counts()


# In[50]:


feat_info.describe()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[51]:


cat_col = feat_info.loc[feat_info['type'] == 'categorical']

cat_col


# In[52]:


cat_col.shape[0]


# In[53]:


cat_col['attribute'].values


# In[54]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

cat_col_binary = []
cat_col_less = []
cat_col_multi = []

for i in cat_col['attribute'].values:
    if azdias[i].nunique() == 2:
        cat_col_binary.append(i)
    elif azdias[i].nunique() < 2:
        cat_col_less.append(i)
    else : 
        cat_col_multi.append(i)
        
    


# In[55]:


cat_col_binary


# In[56]:


cat_col_less # there are no columns with empty datasets


# In[57]:


#Printing the binary columns in categorical feature type

for i in cat_col_binary:
    print(azdias[i].value_counts())


# In[58]:


# Encoding all the binary columns for the categorical feature type

azdias['ANREDE_KZ'].replace([2,1], [1,0], inplace=True)
azdias['VERS_TYP'].replace([2,1], [1,0], inplace=True)
azdias['OST_WEST_KZ'].replace(['W','O'], [1,0], inplace=True)


# In[59]:


for i in cat_col_binary:
    print(azdias[i].value_counts())


# In[60]:


# Re-encode categorical variable(s) to be kept in the analysis.
# All multiple occurances for categorical are dummied

azdias = pd.get_dummies(azdias, columns= cat_col_multi)


# In[61]:


azdias.shape


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# Categorical feature types are deep dived in this encoding process. We are initially analyzing all the column names that come under to the categorical feature type. 
# 
# There seems to be 18 columns out of the selected 81 to be under this list. 
# 
# Each column is checked for binary and multiple extries. There are columns that have binary entries out of which 2 are already in 0 & 1. Other 3 columns are converted to 0s and 1s. 
# 
# All multiple entries in categorical feature type are replaced with dummy variables.

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[62]:


mix_col = feat_info.loc[feat_info['type'] == 'mixed']

mix_col


# In[63]:


feat_info['information_level'].value_counts()


# In[64]:



azdias['PRAEGENDE_JUGENDJAHRE'].head()


# In[65]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

azdias['PRAEGENDE_JUGENDJAHRE'].value_counts(ascending=True)


# In[66]:


mapping_gen_dict = {0: [1, 2], 1: [3, 4], 2 : [5, 6, 7], 3: [8, 9], 4: [10, 11, 12, 13], 5 : [14, 15]} # Categorizing by decades

def map_gen(x):
    try:
        for key, array in mapping_gen_dict.items():
            if x in array:
                return key
    except ValueError:
        return np.nan
    
# Map movement 
mapping_movement = [1, 3, 5, 8, 10, 12, 14] # 1 is mainstream & 0 is avant garde

def map_mov(x):
    try:
        if x in mapping_movement:
            return 1
        else:
            return 0
    except ValueError:
        return np.nan


# In[67]:


# Create generation column
azdias['PRAEGENDE_JUGENDJAHRE_decade'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(map_gen)

# Create movement column
azdias['PRAEGENDE_JUGENDJAHRE_movement'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(map_mov)


# In[68]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
azdias['CAMEO_INTL_2015'].head()


# In[69]:


azdias['CAMEO_INTL_2015'].value_counts(ascending=True)


# In[70]:


# Map wealth 
def map_wealth(i):
    # Check of nan first, or it will convert nan to string 'nan'
    if pd.isnull(i):
        return np.nan
    else:
        return int(str(i)[0])

# Map life stage
def map_lifestage(i):
    if pd.isnull(i):
        return np.nan
    else:
        return int(str(i)[1])


# In[71]:


# Create wealth column
azdias['CAMEO_INTL_2015_wealth'] = azdias['CAMEO_INTL_2015'].apply(map_wealth)

#Create life stage column
azdias['CAMEO_INTL_2015_lifestage'] = azdias['CAMEO_INTL_2015'].apply(map_lifestage)


# In[72]:


# Check
azdias['CAMEO_INTL_2015_wealth'].value_counts()


# In[73]:


azdias['CAMEO_INTL_2015_lifestage'].value_counts()


# In[74]:


azdias = azdias.drop(['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015'], axis=1) # Dropping original after split up


# In[75]:


azdias.shape


# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[76]:


#Life stage, rough scale is categorized

azdias['LP_LEBENSPHASE_GROB'].value_counts() 


# In[77]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

mapping_size = [1, 2, 3, 4, 5, 6, 7] # 1 is single

def map_size(x):
    try:
        if x in mapping_size:
            return 1
        else:
            return 0
    except ValueError:
        return np.nan

mapping_income = [3, 5, 8, 11, 12] # 1 is high

def map_income(x):
    try:
        if x in mapping_income:
            return 1
        else:
            return 0
    except ValueError:
        return np.nan


# In[78]:


# Create generation column
azdias['LP_LEBENSPHASE_GROB_size'] = azdias['LP_LEBENSPHASE_GROB'].apply(map_size)

# Create movement column
azdias['LP_LEBENSPHASE_GROB_income'] = azdias['LP_LEBENSPHASE_GROB'].apply(map_income)


# In[79]:


azdias = azdias.drop(['LP_LEBENSPHASE_GROB'], axis=1) # Dropping original after split up


# In[80]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

np.unique(azdias.dtypes.values)


# In[81]:


azdias.shape


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[82]:


# NOTE I HAVE COMBINED ALL ABOVE TASKS TO THIS FUNCTION MODE

def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    
    df_nan = convert_to_nan(df)
    
    # remove selected columns and rows, ...
    
    df_condensed = df_nan.drop(['TITEL_KZ','AGER_TYP','KK_KUNDENTYP','KBA05_BAUMAX'],axis = 1)

    nan_row = df_condensed.isnull().sum(axis=1)
    
    rows_missing = nan_row[nan_row > 25]

    df_missing = df.iloc[rows_missing.index]

    df_optimized = df_condensed[~df_condensed.index.isin(rows_missing.index)]
    
    df = df_optimized
    
    df['ANREDE_KZ'].replace([2,1], [1,0], inplace=True)
    df['VERS_TYP'].replace([2,1], [1,0], inplace=True)
    df['OST_WEST_KZ'].replace(['W','O'], [1,0], inplace=True)

    df = pd.get_dummies(df, columns= cat_col_multi)
    
    # select, re-encode, and engineer column values.
    # Create generation column
    df['PRAEGENDE_JUGENDJAHRE_decade'] = df['PRAEGENDE_JUGENDJAHRE'].apply(map_gen)

    # Create movement column
    df['PRAEGENDE_JUGENDJAHRE_movement'] = df['PRAEGENDE_JUGENDJAHRE'].apply(map_mov)

    # Create wealth column
    df['CAMEO_INTL_2015_wealth'] = df['CAMEO_INTL_2015'].apply(map_wealth)

    #Create life stage column
    df['CAMEO_INTL_2015_lifestage'] = df['CAMEO_INTL_2015'].apply(map_lifestage)

    # Create generation column
    df['LP_LEBENSPHASE_GROB_size'] = df['LP_LEBENSPHASE_GROB'].apply(map_size)

    # Create movement column
    df['LP_LEBENSPHASE_GROB_income'] = df['LP_LEBENSPHASE_GROB'].apply(map_income)
    
    df = df.drop(['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015', 'LP_LEBENSPHASE_GROB'], axis=1) # Dropping original after split up
    
    # Return the cleaned dataframe.
    return df,df_missing
    


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[83]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean')

imputed_azdias = imputer.fit_transform(azdias)


# In[84]:


# Apply feature scaling to the general population demographics data.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()

standardized_azdias = scaler.fit_transform(imputed_azdias)


# ### Discussion 2.1: Apply Feature Scaling
# 
# In this section , we are ensuring all missing datas are replaced by Nan using imputer functions. 
# 
# Second portion was utilizing standard scaler function to standardize data using z score method in order to perform PCAs.The standardized values are saved in a separate dataframe.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[85]:


# Apply PCA to the data.
from sklearn.decomposition import PCA

n_components = int(np.round(feat_info.shape[0]*0.8)) # 80% of all features


#Copy for later use
azdias_copy = azdias.copy()


pca = PCA(n_components)
azdias_pca = pca.fit_transform(standardized_azdias)


# In[86]:


azdias_copy.shape


# In[87]:


# Investigate the variance accounted for by each principal component.
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(24, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=8)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    

scree_plot(pca)


# In[88]:


a = pca.explained_variance_ratio_
s = 0.0

for i in range (a.shape[0]):
    s = s + a[i]
    if s >= 0.65:
        break
    
print('The features to be used for optimized PCA : ', i)
    


# In[89]:


# Re-apply PCA to the data while selecting for number of components to retain.

n_components_retain = i

pca = PCA(n_components_retain)
azdias_pca_final = pca.fit_transform(standardized_azdias)


scree_plot(pca)


# In[90]:


azdias_pca_final.shape


# In[91]:


pca.components_


# In[92]:


pca.components_.shape


# In[93]:


pca.explained_variance_ratio_


# In[94]:


pca.singular_values_


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# Initial PCA analysis was performed for about 80% of features were used.
# 
# Engaged for loop to see how many components were need to gain atleast 65% of the variance ratio. Found that we will perform the optimized PCA for about 56 features. 

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[95]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

def plot_pca(data, pca, n_comp):
    
    comp = pd.DataFrame(np.round(pca.components_, 2), columns = data.keys()).iloc[n_comp-1]
    comp.sort_values(ascending=False, inplace=True)
    #print(comp.shape)
    comp = pd.concat([comp.head(3), comp.tail(3)])
    #print(comp)
    comp.plot(kind='barh', title='Component ' + str(n_comp))
    ax = plt.gca()
    ax.grid(linewidth='0.25', alpha=1)
    plt.show()
    


# In[96]:


plot_pca(azdias, pca, 1)


# In[97]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

plot_pca(azdias, pca, 2)


# In[98]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

plot_pca(azdias, pca, 3)


# ### Discussion 2.3: Interpret Principal Components
# 
# The first principal component has a positive association with:
# 
#     1. LP_STATUS_GROB_1: Social status, low-income earners (binary)
#     2. HH_EINKOMMEN_SCORE: Estimated household net income (higher is lower income)
#     3. CAMEO_INTL_2015_wealth: Household wealth (higher is lower income)
# 
# and a negative association with:
# 
#     1. MOBI_REGIO: Movement patterns (higher is lower movement)
#     2. KBA05_ANTG1: Number of 1-2 family houses in the microcell (higher is higher share of 1-2 family homes in cell)
#     3. FINANZ_MINIMALIST: Financial topology, low financial interest (higher is lower topology)
# 
# It appears that the first component is related to the indiviudal financial status, movement, and share of 1-2 family homes.
# 
# The second principal component has a positive association with:
#     1. ALTERSKATEGORIE_GROB: Estimated age (higher is older)
#     2. FINANZ_VORSORGER: Financial typology, be prepared (higher is lower topology)
#     3. ZABEOTYP_3: Energy consumption, fair supplied (binary)
# And a negative association with:
#     1. PRAEGENDE_JUGENDJAHRE_decade: Decade of movement of person's youth (highest is 90s)
#     2. FINANZ_SPARER: Financial typology, money-saver (higher is lower topology)
#     3. FINANZ_UNAUFFAELLIGER: Financial typology, inconspicuous (higher is lower topology)
# It seems like the second component is linked to age, decade of movement, and financial savings.
# 
# The third principal component has a positive association with:
#     1. SEMIO_VERT: Personality typology, dreamful (higher is lower affinity)
#     2. SEMIO_FAM: Personality typology, family-minded (higher is lower affinity)
#     3. SEMIO_SOZ: Personality typology, socially-minded (higher is lower affinity)
# And a negative association with:
#     1. ANREDE_KZ: Gender (2 is female)
#     2. SEMIO_KAEM: Personality typology, combative attitude (higher is lower affinity)
#     3. SEMIO_DOM: Personality typology, dominant-minded (higher is lower affinity)
# 
# It looks like the third component is related to the personaly traits and gender. 
# 

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[99]:


# 80% of clustering features is available from 20 % of top PCAs

azdias_reduced_pca = azdias_pca_final[np.random.choice(azdias_pca_final.shape[0], int(azdias_pca_final.shape[0]*0.2), replace=False)]


# In[100]:


# Over a number of different cluster counts...


    # run k-means clustering on the data and...
    # compute the average within-cluster distances.
scores = []

for i in range(10,30):
    
    k_means = KMeans(n_clusters = i, random_state = 42).fit(azdias_reduced_pca)
    scores.append(abs(k_means.score(azdias_reduced_pca)))

c = list(range(10,30))
plt.plot(c, scores)
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE Vs K');
    
    
    
    


# In[101]:



scores = []

for i in range(20,40):
    
    k_means = KMeans(n_clusters = i, random_state = 42).fit(azdias_reduced_pca)
    scores.append(abs(k_means.score(azdias_reduced_pca)))


    


# In[102]:


c = list(range(20,40))
plt.plot(c, scores)
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE Vs K');
    


# 
# Utilizing 28 as the no of clusters using elbow method

# In[103]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
kmeans = KMeans(28).fit(azdias_pca_final)

kmeans_labels = kmeans.predict(azdias_pca_final)


# In[104]:


kmeans_labels


# ### Discussion 3.1: Apply Clustering to General Population
# 
# We start by initially applying the 80-20 rule, a random 20% sample of the PCA componenets may have enough information to identify optimum no. of clusters. 
# 
# Firstly, reduced PCA dataframe we are performing the kmeans algorithm & average log likelyhood score for the PCA algorithm. Initial graph was plotted for SSE scores Vs range of 10 to 30 clusters. 
# 
# Seccondly, reduced PCA dataframe we are performing the kmeans algorithm & average log likelyhood score for the PCA algorithm. Initial graph was plotted for SSE scores Vs range of 20 to 40 clusters. 
# 
# I have choosed 28 clusters since the SSE scores are not changing much after 28. 
# 
# The Kmeans algorithm is re-ran for a 28 clusters and the file is predicted on the original data and saved in kmeans_labels.
# 
# 

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[105]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')


# In[106]:


customers.head()


# In[107]:


customers.shape


# In[108]:


customers.describe().T


# In[109]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

features_customers, customers_missing  = clean_data(customers)


# In[110]:


features_customers.head()


# In[111]:


features_customers.shape


# In[112]:


kmeans_labels.shape


# In[113]:


list(set(azdias.columns) - set(features_customers))


# In[114]:


customers.shape[0]


# In[115]:



customers_extended = customers.copy()
customers_extended = pd.concat([customers_extended, customers_extended.iloc[-1:]], ignore_index=True)


# In[116]:


print(customers_extended.shape[0]) # sanity check
customers_extended.tail(3)


# In[117]:


customers_extended.loc[191652,'GEBAEUDETYP'] = 5.0

features_customers, customers_many_missing  = clean_data(customers_extended)

features_customers.drop([191652], inplace=True)

features_customers.tail(3)


# In[118]:


imputed_customers = imputer.transform(features_customers)

standardized_customers = scaler.transform(imputed_customers)

pca_customers = pca.transform(standardized_customers)

kmeans_customers = kmeans.predict(pca_customers)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[119]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

azdias_missing_array = np.full((azdias_missing.shape[0],), -1)

kmeans_all_labels = np.concatenate([kmeans_labels, azdias_missing_array])

customers_missing_array = np.full((customers_many_missing.shape[0],), -1)

kmeans_all_customers = np.concatenate([kmeans_customers, customers_missing_array])


# In[120]:


dict_data = {'proportion': pd.Series(kmeans_all_labels).value_counts(normalize=True, sort=False), 
          'source': 'general'}

general_proportions = pd.DataFrame(dict_data)

# Proportions for customer data
dict_data = {'proportion': pd.Series(kmeans_all_customers).value_counts(normalize=True, sort=False), 
          'source': 'customer'}

customer_proportions = pd.DataFrame(dict_data)

# Concatenate proportions
total_proportions = pd.concat([general_proportions, customer_proportions])


# In[121]:


fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(ax=ax, x=total_proportions.index, y = total_proportions.proportion, hue=total_proportions.source)
ax.set_xlabel('cluster')
ax.set_title('Proportions per cluster for general vs customer populations');


# In[122]:


# Check difference in cluster proportion for general vs customer populations
diff_customer_proportions = customer_proportions['proportion'] - general_proportions['proportion']
diff_customer_proportions.sort_values(ascending=False, inplace=True)
print('over-represented')
print(diff_customer_proportions[:3])
print('\nunder-represented')
print(diff_customer_proportions[-3:])


# In[123]:


# Impute nans
imputer = Imputer(strategy='median')
imputed_azdias_copy = imputer.fit_transform(azdias_copy)

#for below comparison

scaler = StandardScaler()
standardized_azdias_copy = scaler.fit_transform(imputed_azdias_copy)
pca = PCA(100)
pca_azdias_copy = pca.fit_transform(standardized_azdias_copy)


# In[134]:


def plot_cluster_demographics(k, pca_features, kmeans_labels):
    pca_cluster = pca_features[kmeans_labels == k]

    print('cluster', k, 'accounts for', np.round(pca_cluster.shape[0]*100/azdias_copy.shape[0],3), '% of population')

    standardized_features = pca.inverse_transform(pca_cluster)
    features_cluster1 = scaler.inverse_transform(standardized_features)

    features_cluster1 = pd.DataFrame(np.round(features_cluster1), columns = azdias.columns)

    fig, axs = plt.subplots(2,3, figsize=(18,8))
    sns.countplot(features_cluster1['HH_EINKOMMEN_SCORE'], ax = axs[0,0], color='#9b59b6')
    sns.countplot(features_cluster1['CAMEO_INTL_2015_wealth'], ax = axs[0,1], color='#9b59b6')
    sns.countplot(features_cluster1['KBA05_ANTG1'], ax = axs[0,2], color='#9b59b6')
    sns.countplot(features_cluster1['ALTERSKATEGORIE_GROB'], ax = axs[1,0], color='#9b59b6')
    sns.countplot(features_cluster1['ANREDE_KZ'], ax = axs[1,1], color='#9b59b6')
    sns.countplot(features_cluster1['MOBI_REGIO'], ax = axs[1,2], color='#9b59b6')
    plt.show();


# In[135]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?

plot_cluster_demographics(k=9, pca_features=pca_azdias_copy, kmeans_labels=kmeans_labels)


# In[136]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?

plot_cluster_demographics(k=26, pca_features=pca_azdias_copy, kmeans_labels=kmeans_labels)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# Compared population clusters that were popular and unpopular with the mail order company, evaluting fields related to income, housing, age, gender, and movement patterns.
# 
# The popular cluster contained in general individuals with average income, prosperous households, with a high number of 1-2 family houses in the microcell, in the 46 to 60 years old range (>60 years is also close), equally male or female, and with low movement.
# 
# The unpopular cluster contained in general people with high income, prosperous households, with a high number of 1-2 family houses in the microcell, 46 - 60 years old, mostly male, and with low movement.
# 

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




