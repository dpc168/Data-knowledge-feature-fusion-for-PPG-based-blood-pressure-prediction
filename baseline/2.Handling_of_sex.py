
import pandas as pd

# Convert and save directly
df = pd.read_excel('4_Gaussian_features_BP_with_baseline.xlsx')
df['Sex(M/F)'] = df['Sex(M/F)'].map({'Male': 0, 'Female': 1})
df.to_excel('4_Gaussian_features_BP_with_baseline.xlsx', index=False)
print("Gender data conversion completed")
