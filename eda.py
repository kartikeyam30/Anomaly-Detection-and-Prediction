import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load the dataset from 'light.csv'
df = pd.read_csv('light.csv')

# %%
# Drop columns that are completely empty to clean up the dataset
df.dropna(axis=1, how='all', inplace=True)

# Convert the 'time' column to datetime format and set it as the index of the dataframe
df.time = pd.to_datetime(df.time)
df.set_index('time', inplace=True)

# %% Ordering columns

df = df[['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10',
         'VA', 'W', 'channel', 'irms', 'nid', 'vfreq', 'vh1', 'vrms', 'thing_type']]

# %% Cleaning data

pattern = r'\((-?\d+\.\d+)([+-]\d+\.\d+)j\)'
matches = df.iloc[:, :10].str.extract(pattern)

df.iloc[:, :10] = df.iloc[:, :10].str.replace('(', '').str.replace(')', '')

h_real = df.iloc[:, :10].apply(lambda x: np.real(x))
h_imag = df.iloc[:, :10].apply(lambda x: np.imag(x))

sc = StandardScaler()
h_real = sc.fit_transform(h_real)
h_imag = sc.fit_transform(h_imag)

complex_ = h_real + 1j * h_imag
df_h = pd.DataFrame(complex_, index=df.index, columns=df.columns[:10])
df.iloc[:, :10] = df_h

df = df[['channel', 'nid', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10',
         'VA', 'W', 'irms', 'vfreq', 'vh1', 'vrms']]

df.iloc[:, 13:] = sc.fit_transform(df.iloc[:, 13:])

# %% Labelling by rules

df['Label'] = 0
df.Label = np.where(df.W/df.VA < (1/1.5), 1, 0)
df.Label = np.where(df.W/df.VA < 0.3, 2, df.Label)
df.Label = np.where(df.W/df.VA < 0, -1, df.Label)
df['1/PF'] = df.W/df.VA

df_normal = df[(df.channel == 1) & (df.nid == '1G34PC')]

# %% Analysing assets and plotting their graphs for visual analysis

all_combos = df.groupby(['channel', 'nid', 'thing_type']).size().sort_values(ascending=False).reset_index()

df_summary = df.describe()

summaries = []
sampled = []
for i in range(len(all_combos)):
    print('Started ' + str(i) + ' out of 44')
    df_temp_W = df[(df.channel == all_combos.channel[i]) & (df.nid == all_combos.nid[i])]
    timestamp = [dt.replace(tzinfo=None) for dt in pd.to_datetime(df_temp_W.index)]
    time_diff = (timestamp[-1] - timestamp[0]).total_seconds()/3600

    df_s = df_temp_W.describe()
    df_s = (df_summary-df_s)*100/df_summary

    summaries.append(df_s)

    df_temp_W.set_index(pd.Series(timestamp, name='time'), inplace=True)
    df_temp_W_1h = df_temp_W.resample('0.5H').mean()
    df_temp_W_1h['RollingW'] = df_temp_W_1h.W.rolling(5).mean()
    df_temp_W_1h['thing_type'] = df_temp_W['thing_type'].unique()[0]
    df_temp_W_1h['nid'] = df_temp_W['nid'].unique()[0]
    sampled.append(df_temp_W_1h)
    filename = str(all_combos.channel[i]) + '_' + all_combos.nid[i] + '.jpg'

    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(160, 40))
    ax[0].plot(df_temp_W_1h.W, color='r')
    ax[0].plot(df_temp_W_1h.VA, color='b')
    ax[1].plot(df_temp_W_1h.irms, color='g')
    ax[4].plot(df_temp_W_1h.Hi1, label='Hi1')
    ax[4].plot(df_temp_W_1h.Hi7, label='Hi7')
    ax[2].plot(df_temp_W_1h.Hi2, label='Hi2')
    ax[2].plot(df_temp_W_1h.Hi3, label='Hi3')
    ax[5].plot(df_temp_W_1h.Hq1, label='Hq1')
    ax[5].plot(df_temp_W_1h.Hq7, label='Hq7')
    ax[3].plot(df_temp_W_1h.Hq2, label='Hq2')
    ax[3].plot(df_temp_W_1h.Hq3, label='Hq3')
    ax[5].set_xticks(df_temp_W_1h.resample('1D').mean().index)
    ax[5].tick_params(axis='x', rotation=90)
    ax[2].legend(title='Values')
    ax[3].legend(title='Values')
    ax[4].legend(title='Values')
    ax[5].legend(title='Values')
    plt.tight_layout()
    plt.savefig('Graph-light/'+filename)


# %% Creating Sections of all unique assets, separating large periods of null values as noticed in graphs

i = 0
all_indoor = all_combos[all_combos.thing_type == 'Indoor Lighting']

indoor_final = []
indoor_final_df = []
for i in all_indoor.index:
    row_name = all_indoor.loc[i]
    df_temp = sampled[i]

    df_temp.Label = np.where(df_temp.W/df_temp.VA < (1/1.5), 1, 0)
    df_temp.Label = np.where(df_temp.W/df_temp.VA < 0.3, 2, df_temp.Label)
    df_temp.Label = np.where(df_temp.W/df_temp.VA < 0, -1, df_temp.Label)
    df_temp["1/PF"] = df_temp.W/df_temp.VA

    df_temp.reset_index(inplace=True)
    consecutive_missing_count = 0
    df_count = 0
    threshold = 12
    current_dataframe = None
    for _, row in df_temp.iterrows():
        if all(row[1:-3].isna()):
            consecutive_missing_count += 1
        else:
            consecutive_missing_count = 0

        if consecutive_missing_count >= threshold:
            # Split the DataFrame and reset consecutive_missing_count
            if current_dataframe is not None:
                indoor_final.append(str(row_name.channel) + '_' + row_name.nid + '_' + str(df_count))
                indoor_final_df.append(current_dataframe.iloc[:-4].set_index('time'))
                df_count += 1
            current_dataframe = None

        else:
            if current_dataframe is None:
                current_dataframe = pd.DataFrame(columns=df.columns)
            current_dataframe = current_dataframe.append(row, ignore_index=True)
    indoor_final.append(str(row_name.channel) + '_' + row_name.nid + '_' + str(df_count))
    indoor_final_df.append(current_dataframe.iloc[:-4].set_index('time'))

# %% Saving the cleaned Light Dataset

for df in indoor_final_df:
    df.index = [dt.replace(tzinfo=None) for dt in df.index]

writer = pd.ExcelWriter('all_light.xlsx', mode='w')
for i in range(len(indoor_final)):
    indoor_final_df[i].to_excel(writer, sheet_name=indoor_final[i])
writer.close()


# %% Analysing all new light data

df_all = pd.concat(sampled, ignore_index=False)
df_all.reset_index(inplace=True)
df_all.loc[df_all.time.isna(), 'time'] = df_all[df_all.time.isna()]['index']
df_all.index = df_all.time
df_all.drop(['time', 'index', 'RollingW', '1/PF'], axis=1, inplace=True)

fig, axs = plt.subplots(nrows=11, ncols=4, figsize=(500, 150))

i = 0
j = 0
for (channel, nid), group in df_all.groupby(['channel', 'nid']):
    axs[j, i].plot(group.index, group['W'], label=f'Channel {channel}, NID {nid}')
    axs[j, i].set_xlabel('Time')
    axs[j, i].set_ylabel('Metric')
    axs[j, i].legend()
    axs[j, i].set_xticks(ticks=group.index, rotation=45)
    # Ensure labels and title fit
    if i < 3:
        i += 1
    else:
        j += 1
        i = 0

plt.title('Lighting Data Across Channels and NIDs')
plt.tight_layout()
plt.show()
plt.savefig('All Wattage.jpg')
