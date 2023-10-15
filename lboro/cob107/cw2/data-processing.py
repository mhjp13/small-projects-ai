# Library imports
import numpy as np
import pandas as pd

# Reading orginal dataset into excel file
original_df = pd.read_excel('Original-River-Data.xlsx', usecols='A:I', skiprows=1)
river_data = original_df.copy()

# Renaming Headers
new_columns = {'Unnamed: 0': 'Date'}
new_columns.update({col: f"{col} MDF (Cumecs)" for col in river_data.columns[1:5]})
new_columns.update({col: f"{col} DRT (mm)" for col in river_data.columns[5:]})

river_data.rename(
    columns=new_columns, 
    inplace=True
)

# Converting non-numeric columns to numeric columns
river_data['Skip Bridge MDF (Cumecs)'] = pd.to_numeric(river_data['Skip Bridge MDF (Cumecs)'], errors='coerce')
river_data['Skelton MDF (Cumecs)'] = pd.to_numeric(river_data['Skelton MDF (Cumecs)'], errors='coerce')
river_data['East Cowton DRT (mm)'] = pd.to_numeric(river_data['East Cowton DRT (mm)'], errors='coerce')
river_data['Date'] = pd.to_datetime(river_data['Date'], errors='coerce')

# Dropping rows with at least 1 null value
flow_cols = list(river_data.columns[1:5])
rain_cols = list(river_data.columns[5:])

null_values = river_data.isna().any(axis=1)
river_data.dropna(how="any", inplace=True)

# Dropping rows with rainfall outliers
rainfall_outliers = river_data[(river_data[rain_cols] > 400).any(1)]
river_data.drop(rainfall_outliers.index, inplace=True)

# Dropping rows with river flow outliers
river_flow_outliers = river_data[(river_data[flow_cols] == 0).any(1)]
river_data.drop(river_flow_outliers.index, inplace=True)

# Exporting cleaned dataset to excel file
export_data = river_data.copy()
export_data["Date"] = export_data["Date"].astype("string")
export_data.to_excel('River-Data-Cleaned.xlsx')

# Lagging data
clean_df = pd.read_excel('River-Data-Cleaned.xlsx')
clean_df.drop(["Unnamed: 0"], axis=1, inplace=True)

lagged_df = pd.DataFrame()
lagged_df["Date"] = clean_df["Date"]
lagged_df[flow_cols[-1]] = clean_df[flow_cols[-1]]

## Lagging rainfall and flow columns by 1 to 3 days
for i in range(3):
    for col in flow_cols:
        col_name = col.replace("(Cumecs)", f"(t-{i+1})")
        lagged_df[col_name] = clean_df[col].shift(i+1)

for i in range(3):
    for col in rain_cols:
        col_name = col.replace("(mm)", f"(t-{i+1})")
        lagged_df[col_name] = clean_df[col].shift(i+1)

## Dropping rows with null values
lagged_df[lagged_df.isna().any(axis=1)]
lagged_df.dropna(how="any", inplace=True)

# Moving averages
moving_avg_df = pd.DataFrame()
moving_avg_df["Date"] = clean_df["Date"]
moving_avg_df[flow_cols[-1]] = clean_df[flow_cols[-1]]

## Creating moving averages of between 3 and 7 days for each numerical column
for i in range(3, 8):
    for col in flow_cols:
        col_name = col.replace("(Cumecs)", f"(MA{i})")
        moving_avg_df[col_name] = clean_df[col].rolling(i).mean()

for i in range(3, 8):
    for col in rain_cols:
        col_name = col.replace("(mm)", f"(MA{i})")
        moving_avg_df[col_name] = clean_df[col].rolling(i).mean()

## Dropping rows with null values
moving_avg_df[moving_avg_df.isna().any(axis=1)]
moving_avg_df.dropna(how="any", inplace=True)

# Lagged moving averages
lagged_ma_df = pd.DataFrame()
lagged_ma_df["Date"] = moving_avg_df["Date"]
lagged_ma_df[flow_cols[-1]] = moving_avg_df[flow_cols[-1]]

## Lagging moving averages by 1 day
## lagging them by more than 1 day results in much weaker correlations
mdf_cols = list(moving_avg_df.columns[2:22])
drt_cols = list(moving_avg_df.columns[22:])

for col in mdf_cols:
    col_name = col + f" (t-1)"
    lagged_ma_df[col_name] = moving_avg_df[col].shift(1)

for col in drt_cols:
    col_name = col + f" (t-1)"
    lagged_ma_df[col_name] = moving_avg_df[col].shift(1)

## Dropping rows with null values
lagged_ma_df[lagged_ma_df.isna().any(axis=1)]
lagged_ma_df.dropna(how="any", inplace=True)

# Weighted moving averages
weighted_ma_df = pd.DataFrame()
weighted_ma_df["Date"] = clean_df["Date"]
weighted_ma_df[flow_cols[-1]] = clean_df[flow_cols[-1]]

## Creating weighted moving averages of between 3 and 7 days 
## for each numerical column
for i in range(3, 8):
    for col in flow_cols:
        col_name = col.replace("(Cumecs)", f"(WMA{i})")
        weighted_ma_df[col_name] = clean_df[col].ewm(span=i).mean()

for i in range(3, 8):
    for col in rain_cols:
        col_name = col.replace("(mm)", f"(WMA{i})")
        weighted_ma_df[col_name] = clean_df[col].ewm(span=i).mean()

# Lagged weighted moving averages
lagged_wma_df = pd.DataFrame()
lagged_wma_df["Date"] = weighted_ma_df["Date"]
lagged_wma_df[flow_cols[-1]] = weighted_ma_df[flow_cols[-1]]

## Lagging weighted moving averages
w_mdf_cols = list(weighted_ma_df.columns[2:22])
w_drt_cols = list(weighted_ma_df.columns[22:])

for col in w_mdf_cols:
    col_name = col + f" (t-1)"
    lagged_wma_df[col_name] = weighted_ma_df[col].shift(1)

for col in w_drt_cols:
    col_name = col + f" (t-1)"
    lagged_wma_df[col_name] = weighted_ma_df[col].shift(1)

## Dropping rows with null values
lagged_wma_df[lagged_wma_df.isna().any(1)]
lagged_wma_df.dropna(how="any", inplace=True)

# Exporting newly created datasets with lags and moving averages
lagged_df.to_excel('River-Data-Lagged.xlsx')
moving_avg_df.to_excel('River-Data-MA.xlsx')
lagged_ma_df.to_excel('River-Data-MA-Lagged.xlsx')
weighted_ma_df.to_excel('River-Data-WMA.xlsx')
lagged_wma_df.to_excel('River-Data-WMA-Lagged.xlsx')

# Utility Functions
## Functions for standardising and unstandardising values
def standardise_columns(df, cols):
    """
    This function works with dataframes to standardise values
    in multiple columns to the range [0.1, 0.9]
    """
    subset_df = df[cols]
    subset_df = 0.8 * ((subset_df - subset_df.min()) / (subset_df.max() - subset_df.min())) + 0.1
    return subset_df

def unstandardise_columns(df, cols, max_val, min_val):
    """
    This function works with numpy arrays to destandardise values
    in multiple columns
    """
    subset_df = df[cols]
    subset_df = ((subset_df - subset_df.min()) / 0.8) * (max_val - min_val) + min_val
    return subset_df

def standardise_value(x, max_val, min_val):
    """
    This function works with numpy arrays to standardise values
    in multiple arrays to the range [0.1, 0.9]
    """
    return 0.8 * ((x - min_val)) / (max_val - min_val) + 0.1

def unstandardise_value(x, max_val, min_val):
    """
    This function works with numpy arrays to destandardise values
    in multiple arrays
    """
    return ((x - 0.1) / 0.8) * (max_val - min_val) + min_val

# Function for building custom feature and target sets from larger datasets
def build_feature_set(*datasets):
    assert len(datasets) > 0, "No data sets entered"
    datasets = list(datasets)
    min_rows = min(d.shape[0] for d in datasets)
    
    for i, ds in enumerate(datasets):
        datasets[i] = ds.truncate(before=ds.shape[0]-min_rows).reset_index()
        datasets[i].drop(["index"], axis=1, inplace=True)
        
    merged_df = datasets[0].iloc[:, :2]
    for ds in datasets:
        merged_df = pd.concat([merged_df, ds.iloc[:, 2:]], axis=1)
    
    merged_cols = list(merged_df.columns)
    selected_cols = []
    
    for i in range(0, len(merged_cols), 2):
        format_str = f"{i+1}) {merged_cols[i]}"
        if i != len(merged_cols) - 1:
            second_part = f"{i+2}) {merged_cols[i+1]}"
            num_spaces = 50 - len(format_str)
            format_str += num_spaces*" " + second_part
        print(format_str)
    
    selected_indices = input("\nSelect columns: ")
    for index in selected_indices.split(","):
        if "-" in index:
            first_i, second_i = index.split("-")
            selected_cols += merged_cols[int(first_i) - 1: int(second_i)]
        else:
            selected_cols.append(merged_cols[int(index) - 1])
    
    return merged_df[selected_cols]