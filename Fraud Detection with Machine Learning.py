###########################
# EEE-CIS Fraud Detection #
###########################

# Importing Libraries
import gc
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 700)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Reduce Memory Usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# Loading Data
train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")


# Fix Column Name
fix_col_name = {testIdCol: trainIdCol for testIdCol, trainIdCol in zip(test_identity.columns, train_identity.columns)}
test_identity.rename(columns=fix_col_name, inplace=True)


# Reduce Memory
train_transaction = reduce_mem_usage(train_transaction)
train_identity = reduce_mem_usage(train_identity)

test_transaction = reduce_mem_usage(test_transaction)
test_identity = reduce_mem_usage(test_identity)


# Merge (transaction - identity)
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')


# Merge (X_train - X_test)
train_test = pd.concat([train, test], ignore_index=True)

print(f'train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')

del train_transaction, train_identity, test_transaction, test_identity;
x = gc.collect()

train_test = train_test.copy()
train = train.copy()
test = test.copy()


# Exploratory Data Analysis (EDA)
def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)

    print("##################### Types #####################")
    print(df.dtypes)

    print("##################### Head #####################")
    print(df.head(head))

    print("##################### NA #####################")
    print(df.isnull().sum())

check_df(train)


# Function of Missing Values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 1)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(train, na_name=False)


## Exploring Target Features - isFraud
train_fraud = train.loc[train['isFraud'] == 1]
train_non_fraud = train.loc[train['isFraud'] == 0]

train['isFraud'].value_counts(normalize=True)


sns.countplot(x="isFraud", data=train).set_title('Distribution of Target')
plt.show()



## Exploring Continuous Features

#### TransactionDT
plt.figure(figsize=(15, 5))
sns.distplot(train["TransactionDT"])
sns.distplot(test["TransactionDT"])
plt.title('train vs test TransactionDT distribution')
plt.show()


plt.figure(figsize=(15, 5))
sns.distplot(train_fraud["TransactionDT"], color='b', label='Fraud')
sns.distplot(train_non_fraud["TransactionDT"], color='r', label='non-Fraud')
plt.title('Fraud vs non-Fraud TransactionDT Distribution')
plt.legend()


##### Feature Extraction Taking the start date ‘2015-04-22’, constructed time variables. In discussions tab you should read an excellent solutions.

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

train_test['New_Date'] = train_test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
train_test['New_Date_YMD'] = train_test['New_Date'].dt.year.astype(str) + '-' + train_test['New_Date'].dt.month.astype(
    str) + '-' + train_test['New_Date'].dt.day.astype(str)
train_test['New_Date_YearMonth'] = train_test['New_Date'].dt.year.astype(str) + '-' + train_test[
    'New_Date'].dt.month.astype(str)
train_test['New_Date_Weekday'] = train_test['New_Date'].dt.dayofweek
train_test['New_Date_Hour'] = train_test['New_Date'].dt.hour
train_test['New_Date_Day'] = train_test['New_Date'].dt.day

fig, ax = plt.subplots(4, 1, figsize=(16, 15))

train_test.groupby('New_Date_Weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
train_test.groupby('New_Date_Hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
train_test.groupby('New_Date_Day')['isFraud'].mean().to_frame().plot.bar(ax=ax[2])
train_test.groupby('New_Date_YearMonth')['isFraud'].mean().to_frame().plot.bar(ax=ax[3])


#### TransactionAmt

# TransactionAmt:
print(pd.concat([train['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index(),
                 train_fraud['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index(),
                 train_non_fraud['TransactionAmt'].quantile([.01, .1, .25, .5, .75, .9, .99]).reset_index()],
                axis=1, keys=['Total', 'Fraud', "No Fraud"]))


print(' Fraud TransactionAmt mean      :  ' + str(train_fraud['TransactionAmt'].mean()))
print(' Non - Fraud TransactionAmt mean:  ' + str(train_non_fraud['TransactionAmt'].mean()))


plt.figure(figsize=(15, 5))
sns.distplot(train_test["TransactionAmt"].apply(np.log))
plt.title('Train - Test TransactionAmt distribution')
plt.show()


plt.figure(figsize=(15, 5))
sns.distplot(train_fraud["TransactionAmt"].apply(np.log), label='Fraud | isFraud = 1')
sns.distplot(train_non_fraud["TransactionAmt"].apply(np.log), label='non-Fraud | isFraud = 0')
plt.title('Fraud vs non-Fraud TransactionAmt distribution')
plt.legend()
plt.show()

train['New_TransactionAmt_Bin'] = pd.qcut(train['TransactionAmt'], 15)
train.groupby('New_TransactionAmt_Bin')[['isFraud']].mean()


#### dist1 & dist2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
train['dist1'].plot(kind='hist', bins=5000, ax=ax1, title='dist1 distribution', logx=True)
train['dist2'].plot(kind='hist', bins=5000, ax=ax2, title='dist2 distribution', logx=True)
plt.show()


## Exploring Categorical Features

### Categorical Features
# ProductCD
# addr1 & addr2
# P_emaildomain & R_emaildomain
# DeviceType
# DeviceInfo
def getCatFeatureDetail(df, cat_cols):
    cat_detail_dict = {}
    for col in cat_cols:
        cat_detail_dict[col] = df[col].nunique()
    cat_detail_df = pd.DataFrame.from_dict(cat_detail_dict, orient='index', columns=['nunique'])
    print('There are ' + str(len(cat_cols)) + ' categorical columns.')
    print(cat_detail_df)


cat_features = ['isFraud', 'ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
all_cat_features = cat_features + [f'card{i}' for i in range(1, 7)] + [f'M{i}' for i in range(1, 10)] + [f'id_{i}' for i
                                                                                                         in
                                                                                                         range(12, 39)]

getCatFeatureDetail(train_test, cat_features)


### ProductCD
def ploting_cnt_amt(df, col, lim=2000):
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)
    total = len(df)

    plt.figure(figsize=(16, 14))
    plt.suptitle(f'{col} Distributions ', fontsize=24)

    plt.subplot(211)
    g = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    gt.set_ylim(0, tmp['Fraud'].max() * 1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    sizes = []
    for p in g.patches:
        height = p.get_height()
        sizes.append(height)
        g.text(p.get_x() + p.get_width() / 2.,
               height + 3,
               '{:1.2f}%'.format(height / total * 100),
               ha="center", fontsize=12)

    g.set_ylim(0, max(sizes) * 1.15)
    plt.show()


ploting_cnt_amt(train, 'ProductCD')


### addr1 - addr2
train['addr1'].value_counts().head(10)
train['addr2'].value_counts().head(10)
train.loc[
    train['addr1'].isin(train['addr1'].value_counts()[train['addr1'].value_counts() <= 5000].index), 'addr1'] = "Others"
train.loc[
    train['addr2'].isin(train['addr2'].value_counts()[train['addr2'].value_counts() <= 50].index), 'addr2'] = "Others"

test.loc[test['addr1'].isin(test.addr1.value_counts()[test['addr1'].value_counts() <= 5000].index), 'addr1'] = "Others"
test.loc[test['addr2'].isin(test.addr2.value_counts()[test['addr2'].value_counts() <= 50].index), 'addr2'] = "Others"

train['addr1'].fillna("NoInf", inplace=True)
test['addr1'].fillna("NoInf", inplace=True)

train['addr2'].fillna("NoInf", inplace=True)
test['addr2'].fillna("NoInf", inplace=True)

ploting_cnt_amt(train, "addr1")
ploting_cnt_amt(train, "addr2")


### P-emaildomain & R-emaildomain

### P-emaildomain
train['P_emaildomain'].value_counts()
train.loc[train['P_emaildomain'].isin(['gmail.com', 'gmail']), 'P_emaildomain'] = 'Google'
train.loc[train['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk',
                                       'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                       'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
train.loc[train['P_emaildomain'].isin(['hotmail.com', 'outlook.com', 'msn.com', 'live.com.mx',
                                       'hotmail.es', 'hotmail.co.uk', 'hotmail.de',
                                       'outlook.es', 'live.com', 'live.fr',
                                       'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
train.loc[train['P_emaildomain'].isin(train['P_emaildomain'] \
                                      .value_counts()[train.P_emaildomain.value_counts() <= 500] \
                                      .index), 'P_emaildomain'] = "Others"
train['P_emaildomain'].fillna("NoInf", inplace=True)

# %%

ploting_cnt_amt(train, 'P_emaildomain')

# %% md

### R-emaildomain




# %%

train['R_emaildomain'].value_counts()

# %%

train.loc[train['R_emaildomain'].isin(['gmail.com', 'gmail']), 'R_emaildomain'] = 'Google'

train.loc[train['R_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk',
                                       'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                       'yahoo.es']), 'R_emaildomain'] = 'Yahoo Mail'
train.loc[train['R_emaildomain'].isin(['hotmail.com', 'outlook.com', 'msn.com', 'live.com.mx',
                                       'hotmail.es', 'hotmail.co.uk', 'hotmail.de',
                                       'outlook.es', 'live.com', 'live.fr',
                                       'hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
train.loc[train['R_emaildomain'].isin(train.R_emaildomain \
                                      .value_counts()[train['R_emaildomain'].value_counts() <= 300] \
                                      .index), 'R_emaildomain'] = "Others"
train['R_emaildomain'].fillna("NoInf", inplace=True)

# %%

ploting_cnt_amt(train, 'R_emaildomain')

# %% md

### DeviceType


# %%

train['DeviceType'].value_counts()

# %%

ploting_cnt_amt(train, 'DeviceType')

# %% md

### DeviceInfo


# %%

train['DeviceInfo'].value_counts()

# %%

train['DeviceInfo'].value_counts().head(10).plot(kind='barh', figsize=(15, 5), title='Top 20 Devices in Train')
plt.show()

# %%

train_test['DeviceInfo'] = train_test['DeviceInfo'].fillna('unknown_device').str.lower()
train_test['DeviceName'] = train_test['DeviceInfo'].str.split('/', expand=True)[0]

train_test.loc[train_test['DeviceName'].str.contains('SM', na=False), 'DeviceName'] = 'Samsung'
train_test.loc[train_test['DeviceName'].str.contains('SAMSUNG', na=False), 'DeviceName'] = 'Samsung'
train_test.loc[train_test['DeviceName'].str.contains('GT-', na=False), 'DeviceName'] = 'Samsung'
train_test.loc[train_test['DeviceName'].str.contains('Moto G', na=False), 'DeviceName'] = 'Motorola'
train_test.loc[train_test['DeviceName'].str.contains('Moto', na=False), 'DeviceName'] = 'Motorola'
train_test.loc[train_test['DeviceName'].str.contains('moto', na=False), 'DeviceName'] = 'Motorola'
train_test.loc[train_test['DeviceName'].str.contains('LG-', na=False), 'DeviceName'] = 'LG'
train_test.loc[train_test['DeviceName'].str.contains('rv:', na=False), 'DeviceName'] = 'RV'
train_test.loc[train_test['DeviceName'].str.contains('HUAWEI', na=False), 'DeviceName'] = 'Huawei'
train_test.loc[train_test['DeviceName'].str.contains('ALE-', na=False), 'DeviceName'] = 'Huawei'
train_test.loc[train_test['DeviceName'].str.contains('-L', na=False), 'DeviceName'] = 'Huawei'
train_test.loc[train_test['DeviceName'].str.contains('Blade', na=False), 'DeviceName'] = 'ZTE'
train_test.loc[train_test['DeviceName'].str.contains('BLADE', na=False), 'DeviceName'] = 'ZTE'
train_test.loc[train_test['DeviceName'].str.contains('Linux', na=False), 'DeviceName'] = 'Linux'
train_test.loc[train_test['DeviceName'].str.contains('XT', na=False), 'DeviceName'] = 'Sony'
train_test.loc[train_test['DeviceName'].str.contains('HTC', na=False), 'DeviceName'] = 'HTC'
train_test.loc[train_test['DeviceName'].str.contains('ASUS', na=False), 'DeviceName'] = 'Asus'

train_test.loc[train_test['DeviceName'].isin(train_test['DeviceName'].value_counts()[train_test[
                                                                                         'DeviceName'].value_counts() < 1000].index), 'DeviceName'] = "Others"

# %%

ploting_cnt_amt(train_test, 'DeviceName')

# %% md

## Exploring Group Features (card, C, D, M, V, id )

## card1-card6



# %%

card_cols = [c for c in train.columns if 'card' in c]
train[card_cols].head()

# %%

train_test[card_cols].isnull().sum()

# %%

for col in card_cols:
    print(col + '  :' + str(train[col].nunique()))

# %%

for col in ['card2', 'card3', 'card4', 'card5', 'card6']:
    train_test[col] = train_test.groupby(['card1'])[col].transform(lambda x: x.mode(dropna=False).iat[0])
    train_test[col].fillna(train_test[col].mode()[0], inplace=True)
    print(col + ' has : ' + str(train_test[col].isnull().sum()) + ' missing values')

# %%

ploting_cnt_amt(train, 'card4')

# %%

ploting_cnt_amt(train, 'card6')

# %% md

### C1-C14

# C1 - C14:

# %%

c_cols = [c for c in train if c[0] == 'C']
train[c_cols].head()

# %%

train[c_cols].describe()

# %%

train[c_cols].quantile([.01, .1, .25, .5, .75, .9, .99])

# %%

# train[train['C6']>118.000]['isFraud'].mean()

for col in c_cols:
    print('\n Fraud ' + col + ' mean    :  ' + str(train_fraud[train_fraud[col] <= 37.00][col].mean()))
    print(' Non - Fraud ' + col + ' mean:  ' + str(train_non_fraud[train_non_fraud[col] <= 37.00][col].mean()))

# %% md

### D1-D15

# %%

d_cols = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14']
train[d_cols].head()

# %%

train[d_cols].describe()

# %%

for col in d_cols:
    plt.figure(figsize=(15, 5))
    plt.scatter(train['TransactionDT'], train[col])
    plt.title(col + ' Vs TransactionDT')
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.show()

# %%

msno.matrix(train[d_cols]);

# %% md

### M1-M9

#M1 - M9:

# %%

m_cols = [c for c in train if c[0] == 'M']
for col in m_cols:
    ploting_cnt_amt(train, col, lim=2500)

# %%

msno.matrix(train[m_cols]);

# %% md

### V1-V339

# Vxxx:

# %%

v_cols = [c for c in train if c[0] == 'V']
train[v_cols].head()

# %%

train[v_cols].describe()

# %%

v_cols = [c for c in train_test if c[0] == 'V']
v_nan_df = train_test[v_cols].isna()
nan_groups = {}

for col in v_cols:
    cur_group = v_nan_df[col].sum()
    try:
        nan_groups[cur_group].append(col)
    except:
        nan_groups[cur_group] = [col]
del v_nan_df;
x = gc.collect()


# %%

def plot_corr(v_cols):
    cols = v_cols + ['TransactionDT']
    plt.figure(figsize=(15, 15))
    sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
    plt.title(v_cols[0] + ' - ' + v_cols[-1], fontsize=14)
    plt.show()


# %%

for k, v in nan_groups.items():
    plot_corr(v)

# %% md

### id1-id38

# id1 - id11


# %%

id_cols = [c for c in train_test if c[:2] == 'id']

id_num_cols = id_cols[:11]
id_cat_cols = id_cols[11:]

# %%

train[id_num_cols].describe()

# %%

for col in id_num_cols:
    print('\n' + col)
    print(' Fraud mean    :  ' + str(train_fraud[col].mean()))
    print(' Non - Fraud mean:  ' + str(train_non_fraud[col].mean()))

# %%

getCatFeatureDetail(train, id_cat_cols)

# %%

for col in ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29']:
    ploting_cnt_amt(train, col, lim=2500)


# %% md

# MODEL TUNING

# %% md

### Context Manager

# %%

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(15, 20))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


# %%

def loading_data():
    print('LOADING DATA')
    train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
    train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

    test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
    test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

    # Fix column name
    fix_col_name = {testIdCol: trainIdCol for testIdCol, trainIdCol in
                    zip(test_identity.columns, train_identity.columns)}
    test_identity.rename(columns=fix_col_name, inplace=True)

    ## Reduce memory
    train_transaction = reduce_mem_usage(train_transaction)
    train_identity = reduce_mem_usage(train_identity)

    test_transaction = reduce_mem_usage(test_transaction)
    test_identity = reduce_mem_usage(test_identity)

    # Merge (transaction-identity)
    train = train_transaction.merge(train_identity, on='TransactionID', how='left')
    test = test_transaction.merge(test_identity, on='TransactionID', how='left')

    # MERGE (X_train - X_test)
    train_test = pd.concat([train, test], ignore_index=True)

    print(f'train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
    print(f'test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')

    del train_transaction, train_identity, test_transaction, test_identity;
    x = gc.collect()
    return train_test


# %%

def processing_data(train_test):
    print('PROCESSING DATA')
    drop_col_list = []

    # TransactionDT
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    train_test['NewDate'] = train_test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    train_test['NewDate_YMD'] = train_test['NewDate'].dt.year.astype(str) + '-' + train_test['NewDate'].dt.month.astype(
        str) + '-' + train_test['NewDate'].dt.day.astype(str)
    train_test['NewDate_YearMonth'] = train_test['NewDate'].dt.year.astype(str) + '-' + train_test[
        'NewDate'].dt.month.astype(str)
    train_test['NewDate_Weekday'] = train_test['NewDate'].dt.dayofweek
    train_test['NewDate_Hour'] = train_test['NewDate'].dt.hour
    train_test['NewDate_Day'] = train_test['NewDate'].dt.day
    drop_col_list.extend(["TransactionDT", "NewDate"])  ## !!!

    # TransactionAMT
    train_test['New_Cents'] = (train_test['TransactionAmt'] - np.floor(train_test['TransactionAmt'])).astype('float32')
    train_test['New_TransactionAmt_Bin'] = pd.qcut(train_test['TransactionAmt'], 15)

    # cardX
    card_cols = [c for c in train_test if c[0:2] == 'ca']
    for col in ['card2', 'card3', 'card4', 'card5', 'card6']:
        train_test[col] = train_test.groupby(['card1'])[col].transform(lambda x: x.mode(dropna=False).iat[0])
        train_test[col].fillna(train_test[col].mode()[0], inplace=True)

    # P_email_domain & R_email_domain
    train_test.loc[train_test['P_emaildomain'].isin(['gmail.com', 'gmail']), 'P_emaildomain'] = 'Google'
    train_test.loc[train_test['P_emaildomain'].isin(
        ['yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk', 'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
         'yahoo.es']), 'P_emaildomain'] = 'Yahoo'
    train_test.loc[train_test['P_emaildomain'].isin(
        ['hotmail.com', 'outlook.com', 'msn.com', 'live.com.mx', 'hotmail.es', 'hotmail.co.uk', 'hotmail.de',
         'outlook.es', 'live.com', 'live.fr', 'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
    train_test.loc[train_test['P_emaildomain'].isin(train_test['P_emaildomain'].value_counts()[train_test[
                                                                                                   'P_emaildomain'].value_counts() <= 500].index), 'P_emaildomain'] = "Others"
    train_test['P_emaildomain'].fillna("Unknown", inplace=True)

    train_test.loc[train_test['R_emaildomain'].isin(['gmail.com', 'gmail']), 'R_emaildomain'] = 'Google'
    train_test.loc[train_test['R_emaildomain'].isin(
        ['yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk', 'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
         'yahoo.es']), 'R_emaildomain'] = 'Yahoo'
    train_test.loc[train_test['R_emaildomain'].isin(
        ['hotmail.com', 'outlook.com', 'msn.com', 'live.com.mx', 'hotmail.es', 'hotmail.co.uk', 'hotmail.de',
         'outlook.es', 'live.com', 'live.fr', 'hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
    train_test.loc[train_test['R_emaildomain'].isin(train_test['R_emaildomain'].value_counts()[train_test[
                                                                                                   'R_emaildomain'].value_counts() <= 300].index), 'R_emaildomain'] = "Others"
    train_test['R_emaildomain'].fillna("Unknown", inplace=True)

    # DeviceInfo
    train_test['DeviceInfo'] = train_test['DeviceInfo'].fillna('unknown_device').str.lower()
    train_test['DeviceInfo'] = train_test['DeviceInfo'].str.split('/', expand=True)[0]

    train_test.loc[train_test['DeviceInfo'].str.contains('SM', na=False), 'DeviceInfo'] = 'Samsung'
    train_test.loc[train_test['DeviceInfo'].str.contains('SAMSUNG', na=False), 'DeviceInfo'] = 'Samsung'
    train_test.loc[train_test['DeviceInfo'].str.contains('GT-', na=False), 'DeviceInfo'] = 'Samsung'
    train_test.loc[train_test['DeviceInfo'].str.contains('Moto G', na=False), 'DeviceInfo'] = 'Motorola'
    train_test.loc[train_test['DeviceInfo'].str.contains('Moto', na=False), 'DeviceInfo'] = 'Motorola'
    train_test.loc[train_test['DeviceInfo'].str.contains('moto', na=False), 'DeviceInfo'] = 'Motorola'
    train_test.loc[train_test['DeviceInfo'].str.contains('LG-', na=False), 'DeviceInfo'] = 'LG'
    train_test.loc[train_test['DeviceInfo'].str.contains('rv:', na=False), 'DeviceInfo'] = 'RV'
    train_test.loc[train_test['DeviceInfo'].str.contains('HUAWEI', na=False), 'DeviceInfo'] = 'Huawei'
    train_test.loc[train_test['DeviceInfo'].str.contains('ALE-', na=False), 'DeviceInfo'] = 'Huawei'
    train_test.loc[train_test['DeviceInfo'].str.contains('-L', na=False), 'DeviceInfo'] = 'Huawei'
    train_test.loc[train_test['DeviceInfo'].str.contains('Blade', na=False), 'DeviceInfo'] = 'ZTE'
    train_test.loc[train_test['DeviceInfo'].str.contains('BLADE', na=False), 'DeviceInfo'] = 'ZTE'
    train_test.loc[train_test['DeviceInfo'].str.contains('Linux', na=False), 'DeviceInfo'] = 'Linux'
    train_test.loc[train_test['DeviceInfo'].str.contains('XT', na=False), 'DeviceInfo'] = 'Sony'
    train_test.loc[train_test['DeviceInfo'].str.contains('HTC', na=False), 'DeviceInfo'] = 'HTC'
    train_test.loc[train_test['DeviceInfo'].str.contains('ASUS', na=False), 'DeviceInfo'] = 'Asus'

    train_test.loc[train_test['DeviceInfo'].isin(train_test['DeviceInfo'].value_counts()[train_test[
                                                                                             'DeviceInfo'].value_counts() < 1000].index), 'DeviceInfo'] = "Others"

    # V1 - V339
    v_cols = [c for c in train_test if c[0] == 'V']
    v_nan_df = train_test[v_cols].isna()
    nan_groups = {}

    for col in v_cols:
        cur_group = v_nan_df[col].sum()
        try:
            nan_groups[cur_group].append(col)
        except:
            nan_groups[cur_group] = [col]
    del v_nan_df;
    x = gc.collect()

    for nan_cnt, v_group in nan_groups.items():
        train_test['New_v_group_' + str(nan_cnt) + '_nulls'] = nan_cnt
        sc = preprocessing.MinMaxScaler()
        pca = PCA(n_components=2)
        v_group_pca = pca.fit_transform(sc.fit_transform(train_test[v_group].fillna(-1)))
        train_test['New_v_group_' + str(nan_cnt) + '_pca0'] = v_group_pca[:, 0]
        train_test['New_v_group_' + str(nan_cnt) + '_pca1'] = v_group_pca[:, 1]

    drop_col_list.extend(v_cols)

    print('CREATING NEW FEATURES')

    train_test['New_card1_card2'] = train_test['card1'].astype(str) + '_' + train_test['card2'].astype(str)
    train_test['New_addr1_addr2'] = train_test['addr1'].astype(str) + '_' + train_test['addr2'].astype(str)
    train_test['New_card1_card2_addr1_addr2'] = train_test['card1'].astype(str) + '_' + train_test['card2'].astype(
        str) + '_' + train_test['addr1'].astype(str) + '_' + train_test['addr2'].astype(str)

    train_test['New_P_emaildomain_addr1'] = train_test['P_emaildomain'] + '_' + train_test['addr1'].astype(str)
    train_test['New_R_emaildomain_addr2'] = train_test['R_emaildomain'] + '_' + train_test['addr2'].astype(str)

    # Aggregation features
    train_test['New_TransactionAmt_to_mean_card1'] = train_test['TransactionAmt'] / train_test.groupby(['card1'])[
        'TransactionAmt'].transform('mean')
    train_test['New_TransactionAmt_to_mean_card4'] = train_test['TransactionAmt'] / train_test.groupby(['card4'])[
        'TransactionAmt'].transform('mean')
    train_test['New_TransactionAmt_to_std_card1'] = train_test['TransactionAmt'] / train_test.groupby(['card1'])[
        'TransactionAmt'].transform('std')
    train_test['TransactionAmt_to_std_card4'] = train_test['TransactionAmt'] / train_test.groupby(['card4'])[
        'TransactionAmt'].transform('std')

    train_test['New_id_02_to_mean_card1'] = train_test['id_02'] / train_test.groupby(['card1'])['id_02'].transform(
        'mean')
    train_test['New_id_02_to_mean_card4'] = train_test['id_02'] / train_test.groupby(['card4'])['id_02'].transform(
        'mean')
    train_test['New_id_02_to_std_card1'] = train_test['id_02'] / train_test.groupby(['card1'])['id_02'].transform('std')
    train_test['New_id_02_to_std_card4'] = train_test['id_02'] / train_test.groupby(['card4'])['id_02'].transform('std')

    train_test['New_D15_to_mean_card1'] = train_test['D15'] / train_test.groupby(['card1'])['D15'].transform('mean')
    train_test['New_D15_to_mean_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('mean')
    train_test['New_D15_to_std_card1'] = train_test['D15'] / train_test.groupby(['card1'])['D15'].transform('std')
    train_test['New_D15_to_std_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('std')

    train_test['New_D15_to_mean_addr1'] = train_test['D15'] / train_test.groupby(['addr1'])['D15'].transform('mean')
    train_test['New_D15_to_mean_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('mean')
    train_test['New_D15_to_std_addr1'] = train_test['D15'] / train_test.groupby(['addr1'])['D15'].transform('std')
    train_test['New_D15_to_std_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('std')

    drop_col_list.extend(card_cols)

    # Frequency Encoding
    fe_col_list = ["New_TransactionAmt_Bin", 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceType',
                   'DeviceInfo'] + [c for c in train_test if c[0] == 'M']
    for col in fe_col_list:
        vc = train_test[col].value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = 'New_' + col + '_FE'
        train_test[nm] = train_test[col].map(vc)
        train_test[nm] = train_test[nm].astype('float32')

    print('DROPING UNNECESSARY FEATURES')
    train_test = train_test.drop(drop_col_list, axis=1)

    print('APPLYING LABEL ENCODING TO CATEGORICAL FEATURES')
    for col in train_test.columns:
        if train_test[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(train_test[col].astype(str).values))
            train_test[col] = le.transform(list(train_test[col].astype(str).values))

    print('REDUCING MEMORY USAGE')
    train_test = reduce_mem_usage(train_test)

    print('DATA IS READY TO MODELLING')

    return train_test


# %%

def modeling(train_test, target):
    train = train_test[train_test[target].notnull()]
    test = train_test[train_test[target].isnull()]

    folds = KFold(n_splits=10, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])

    feature_importance_df = pd.DataFrame()

    features = [f for f in train.columns if f not in [target, 'TransactionID', 'New_TransactionAmt_Bin', 'NewDate']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[target])):
        start_time = time.time()
        print('Training on fold {}'.format(n_fold + 1))

        X_train, y_train = train[features].iloc[train_idx], train[target].iloc[train_idx]

        X_valid, y_valid = train[features].iloc[valid_idx], train[target].iloc[valid_idx]

        params = {'learning_rate': 0.01,
                  'objective': 'binary',
                  'metric': 'auc',
                  'num_threads': -1,
                  'num_leaves': 256,
                  'verbose': 1,
                  'random_state': 42,
                  'bagging_fraction': 1,
                  'feature_fraction': 0.85}

        clf = LGBMClassifier(**params, n_estimators=1000)  # categorical_feature = LGBM_cat_col_list

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        # y_pred_valid
        oof_preds[valid_idx] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_valid, oof_preds[valid_idx])))

    print('Full AUC score %.6f' % roc_auc_score(train[target], oof_preds))  # y_pred_valid

    test[target] = sub_preds
    test[['TransactionID', target]].to_csv("submission_lightgbm2.csv", index=False)

    display_importances(feature_importance_df)

    return feature_importance_df


# %%

def main():
    with timer("Loading Data"):
        train_test = loading_data()

    with timer("Preprocessing Data"):
        train_test = processing_data(train_test)

    with timer("Modeling"):
        feat_importance = modeling(train_test, 'isFraud')


# %%

if __name__ == "__main__":
    with timer("Full model run"):
        main()

