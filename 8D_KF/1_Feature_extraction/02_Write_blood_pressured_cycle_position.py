import pandas as pd

# Read raw data
raw_ppg_df = pd.read_excel('raw_PPG.xlsx')
result_df = pd.read_excel('01_result.xlsx')

# Extract required columns from raw_PPG
ppg_data = raw_ppg_df[['Name', 'start', 'end', 'Y_S', 'Y_D']]

# Extract required columns from result (except Name)
result_data = result_df[['t1', 'h1', 't2', 'h2', 'Tsys', 't3', 'h3', 'T', 'Tdia']]

# Merge two datasets by Name column
merged_df = pd.concat([ppg_data, result_data], axis=1)

# Rearrange column order to required format
final_df = merged_df[['Name', 'start', 'end', 't1', 'h1', 't2', 'h2', 'Tsys', 't3', 'h3', 'T', 'Tdia', 'Y_S', 'Y_D']]

# Save to new file
final_df.to_excel('8D_KF.xlsx', index=False)

print("File successfully saved as 8D_KF.xlsx")
print(f"Processed {len(final_df)} records in total")
print("\nPreview of first 5 rows:")
print(final_df.head())