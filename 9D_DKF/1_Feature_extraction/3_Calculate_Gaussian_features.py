import pandas as pd

# Read Excel file
file_path = '2.Gaussian_optimal_parameters.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Calculate differences
df['b2-b1'] = df['b2'] - df['b1']
df['b3-b2'] = df['b3'] - df['b2']
df['b3-b1'] = df['b3'] - df['b1']

# Delete original columns (b1, b2, b3, Residual)
columns_to_drop = ['b1', 'b2', 'b3', 'Residual']
df = df.drop(columns=columns_to_drop, errors='ignore')  # `errors='ignore'` avoids errors if columns don't exist

# Save to new Excel file
output_path = '3_Gaussian_features.xlsx'
df.to_excel(output_path, index=False)

print(f"Calculation results saved to: {output_path}")