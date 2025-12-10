import pandas as pd

# Load both Excel files
gaussian_df = pd.read_excel('2_Gaussian_features.xlsx')
features_df = pd.read_excel('raw_PPG.xlsx')

# Extract the Name, Y_S, and Y_D columns from the features file
bp_data = features_df[['Name', 'Y_S', 'Y_D']]

# Merge the BP data with the Gaussian features data based on Name
merged_df = pd.merge(gaussian_df, bp_data, on='Name', how='left')

# Save the updated DataFrame back to the original file or a new file
merged_df.to_excel('3_Gaussian_features_BP.xlsx', index=False)

print("BP values have been successfully added to the Gaussian features file.")