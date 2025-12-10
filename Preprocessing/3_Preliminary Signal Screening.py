import openpyxl
import pandas as pd

# Read the original Excel file
df = pd.read_excel('cycle_data_psqi.xlsx')

# Filter rows where PSQI is greater than 0.6
filtered_df = df[df['PSQI'] > 0.6]

# Save to new Excel file in original order
filtered_df.to_excel('Preliminary_PSQI.xlsx', index=False)

print("Filtering completed, results saved to Preliminary_PSQI.xlsx")