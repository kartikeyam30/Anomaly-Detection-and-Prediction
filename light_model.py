import gc
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import joblib
from itertools import chain
from collections import Counter

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df_map = pd.read_excel('all_light.xlsx', sheet_name=None)

# %% Segmenting into smaller dataframes and performing operations such as scaling and decomposition

segment_size = 500
overlap_size = int(0.8 * segment_size)

segments = []
names = []
for key, df in df_map.items():
    i = 0
    if len(df) >= (1.2 * segment_size):
        for start in range(0, len(df) - segment_size + 1, segment_size - overlap_size):
            segment = df.iloc[start:start + segment_size]
            if len(segment) < 200:
                break
            nid = segment['nid'].unique()
            segment.drop('nid', axis=1, inplace=True)
            segment.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
            segment.set_index('time', inplace=True)
            segment = segment.iloc[:, :27]
            segment.index = [dt.replace(tzinfo=None) for dt in pd.to_datetime(segment.index)]
            segment = segment.resample('0.5H').mean()
            segment.fillna(method='ffill')
            segment['nid'] = nid[0]
            segments.append(segment)
            names.append(key + str(i))
            start += segment_size
            i += 1

# %% Dimensionality reduction by correlation analysis
correlated_pairs = {}

df_final = []
for df in segments:
    df_nid = df.nid.unique()[0]
    df.drop(['nid'], axis=1, inplace=True)
    df = df[['Hi1', 'Hi10', 'Hi2', 'Hi3', 'Hi4', 'Hi5', 'Hi6', 'Hi7', 'Hi8',
             'Hi9', 'Hq1', 'Hq10', 'Hq2', 'Hq3', 'Hq4', 'Hq5', 'Hq6', 'Hq7', 'Hq8',
             'Hq9', 'VA', 'W', 'irms', 'vfreq', 'vh1', 'vrms']]
    df_final.append(df)
    corr_matrix = df.corr()
    # Filter for coefficients greater than 0.8 and less than 1 (to exclude self-correlation)
    highly_correlated = corr_matrix[(corr_matrix > 0.8) & (corr_matrix < 1)]

    # Identify column pairs with high correlation
    for column in highly_correlated.columns:
        correlated_cols = highly_correlated[column].dropna().index.tolist()
        for col in correlated_cols:
            pair = tuple(sorted([column, col]))
            if pair not in correlated_pairs:
                correlated_pairs[pair] = 1
            else:
                correlated_pairs[pair] += 1

threshold = 0.6 * len(segments)
pairs_to_drop = [pair for pair, count in correlated_pairs.items() if count >= threshold]

column_occurrences = {}

# Count occurrences of each column in the pairs to drop
for pair in pairs_to_drop:
    for col in pair:
        if col not in column_occurrences:
            column_occurrences[col] = 1
        else:
            column_occurrences[col] += 1

print(column_occurrences)

# %% Feature Creation

column_occurrences.pop('W', None)
column_occurrences.pop('VA', None)

for df in df_final:
    if 'Hi1' in df.columns:
        df.drop(['Hi1', 'Hq7', 'Hq9', 'Hq3', 'irms', 'vh1'], axis=1, inplace=True)

    roll_depths = [5, 10, 20]
    for col in df.columns:
        df[col] = df[col].interpolate(method='linear', axis=0)
        if col not in ['nid', 'vfreq', 'vrms']:
            for depth in roll_depths:
                df[col + '_diff'] = df[col].diff()
                df[col + '_roll_mean_'+str(depth)] = df[col].rolling(depth).mean()
                df[col + '_roll_min_'+str(depth)] = df[col].rolling(depth).min()
                df[col + '_roll_max_'+str(depth)] = df[col].rolling(depth).max()
                df[col + '_roll_std_' + str(depth)] = df[col].rolling(depth).std()

        if col in ['Hi10', 'Hi2', 'Hi3', 'Hi4', 'Hi5', 'Hi6', 'Hi7', 'Hi8', 'Hi9', 'Hq1',
                   'Hq10', 'Hq2', 'Hq4', 'Hq5', 'Hq6', 'Hq8']:
            series = df.loc[df.index[19]:, col]
            decomposition = seasonal_decompose(series, period=48, model='additive')
            detrended = series - decomposition.trend

            np.fft.fftfreq(series.shape[0] - 19, d=1800)
            transformed = np.fft.fft(detrended)
            magnitude = np.abs(transformed)
            filtered_signal = np.fft.ifft(transformed)

            df[col + 'detrended'] = pd.Series(filtered_signal.real, index=df.index[19:])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    df['PF'] = df.VA / df.W
    df['DOTW'] = df.index.day_of_week
    df['week#'] = df.index.isocalendar().week
    df['VA_lag1'] = df.VA.shift(1)
    df['VA_lag2'] = df.VA.shift(2)
    df['VA_lag3'] = df.VA.shift(3)
    df['VA_lag4'] = df.VA.shift(4)
    df['VA_lag5'] = df.VA.shift(5)

# %% Scaling and Null filling

df_scaled = []
df_all = pd.concat([df.iloc[5:,] for df in df_final], ignore_index=True)
df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
df_all.fillna(method='ffill', axis=0, inplace=True)
sc = MinMaxScaler()
sc.fit(df_all)

joblib.dump(sc, 'all_scaler.bin')

for df in df_final:
    df = df.loc[df.index[5]:]
    df.PF.replace(np.inf, df['PF'].replace([np.inf, -np.inf], np.nan).max(), inplace=True)
    df_s = pd.DataFrame(sc.transform(df), columns=df.columns, index=df.index)
    df_scaled.append(df_s)

# %% Re-dropping based on correlation analysis on new columns

df_all = pd.concat([df[df.columns[:-2]] for df in df_scaled], ignore_index=True)
corr_matrix = df_all.corr()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
highly_correlated_pairs = [(col1, col2) for col1 in upper.columns for col2 in upper.index if
                           upper.at[col2, col1] > 0.9]


all_correlated_columns = list(chain(*highly_correlated_pairs))

frequency_of_columns = Counter(all_correlated_columns)
sorted_columns_by_frequency = [item for item, count in frequency_of_columns.items()]

# Identify groups of correlated columns (dict to store which columns are related to which)
correlated_groups = {}
for col1, col2 in highly_correlated_pairs:
    if col1 in correlated_groups:
        correlated_groups[col1].add(col2)
    elif col2 in correlated_groups:
        correlated_groups[col2].add(col1)
    else:
        correlated_groups[col1] = {col2}

columns_to_drop = set()
for main_col, corr_cols in correlated_groups.items():
    # Get the most frequently occurring column from each group to keep
    most_frequent = max((col for col in [main_col] + list(corr_cols)), key=lambda col: frequency_of_columns[col])
    # Add all other columns to the drop list
    columns_to_drop.update(corr_cols.difference({most_frequent}))

for df in df_scaled:
    df.drop([elem for elem in columns_to_drop
             if elem not in ['VA', 'W', 'VA_lag1', 'VA_lag2', 'VA_lag3', 'VA_lag4', 'VA_lag5']],
            axis=1, inplace=True)

df_reduce = [col for col in df_scaled[0].columns if col not in ['W', 'VA']]

df_all = pd.concat(df_scaled, ignore_index=True)

np.save('raw', np.array([df.values for df in df_scaled]))
np.save('time_stamps', np.array([df.index for df in df_scaled]))

# %% Dimensionality Reduction based on PCA

pca1 = PCA(n_components=0.95)

df_hi = df_all[[col for col in df_all.columns if col.startswith('H')]].values
df_v = df_all[[col for col in df_all.columns if col.startswith('v')]].values
df_wva = df_all[[col for col in df_all.columns if (col.startswith('W_') or col.startswith('VA_'))]].values

df_hi_red = pca1.fit_transform(df_hi)
df_v_red = pca1.fit_transform(df_v)
df_wva_red = pca1.fit_transform(df_wva)

df_reduced = np.concatenate((df_hi_red, df_v_red, df_wva_red,
                             df_all[['DOTW', 'PF', 'week#', 'W', 'VA']].values), axis=1)

df_reduced = df_reduced.reshape((len(names), segment_size-5, df_reduced.shape[1]))

df_pca = []
for i in range(df_reduced.shape[0]):
    df = pd.DataFrame(df_reduced[i], index=df_scaled[i].index)
    df_pca.append(df)

del df_final, segments, segment
gc.collect()
gc.collect()
# %% Splitting into equally segmented samples and saving as X and Y

all = []
X = []
Y = []

for i in range(1, len(df_pca)):
    name_cur = names[i].split('_')
    name_prev = names[i - 1].split('_')
    if name_cur[:2] == name_prev[:2]:
        date_prev = df_pca[i - 1].index[-1]
        if (date_prev in df_pca[i].index) and len(df_pca[i].loc[date_prev:]) > 100:
            X.append(np.array(df_pca[i - 1].values, dtype=np.float32))
            Y.append(np.array(df_pca[i].loc[date_prev + pd.DateOffset(minutes=30):
                                            (date_prev + pd.DateOffset(minutes=30 * int(100)))], dtype=np.float32))
all.append(np.concatenate((X, Y), axis=1, dtype=np.float16))

np.save('X_all', all[0])
np.save('X_data_500', X)
np.save('Y_data_500', Y)
