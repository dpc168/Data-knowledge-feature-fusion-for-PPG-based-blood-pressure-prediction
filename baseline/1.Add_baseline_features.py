import pandas as pd


def merge_ppg_bp_data(gaussian_file, ppg_bp_file, output_file=None):
    """
    Merge demographic information from PPG-BP dataset into Gaussian features dataset

    Parameters:
    gaussian_file: Path to 4_Gaussian_features_BP.xlsx file
    ppg_bp_file: Path to PPG-BP dataset.xlsx file
    output_file: Output file path (optional)

    Returns:
    Merged DataFrame
    """

    # Read Gaussian features data
    gaussian_df = pd.read_excel(gaussian_file, sheet_name='Sheet1')

    # Read PPG-BP data
    ppg_bp_df = pd.read_excel(ppg_bp_file, sheet_name='Name')

    # Extract main ID from Gaussian data's Name column (remove part after underscore)
    gaussian_df['Main_ID'] = gaussian_df['Name'].astype(str).str.split('_').str[0]

    # Extract main ID from PPG-BP data's Name column (already main ID)
    ppg_bp_df['Main_ID'] = ppg_bp_df['Name'].astype(str)

    # Select columns to merge
    columns_to_merge = ['Sex(M/F)', 'Age(year)', 'Height(cm)', 'Weight(kg)']

    # Merge data
    merged_df = gaussian_df.merge(
        ppg_bp_df[['Main_ID'] + columns_to_merge],
        on='Main_ID',
        how='left'
    )

    # Remove temporary Main_ID column
    merged_df = merged_df.drop('Main_ID', axis=1)

    # Save to file if needed
    if output_file:
        merged_df.to_excel(output_file, index=False)
        print(f"Merged data saved to: {output_file}")

    return merged_df


# Usage example
if __name__ == "__main__":
    # Call function
    result_df = merge_ppg_bp_data(
        gaussian_file='4_Gaussian_features_BP.xlsx',
        ppg_bp_file='PPG-BP dataset.xlsx',
        output_file='4_Gaussian_features_BP_with_baseline.xlsx'
    )

    # Display first few rows to verify merge result
    print("First 5 rows of merged data:")
    print(result_df.head())

    # Display data shape
    print(f"\nMerged data shape: {result_df.shape}")

    # Check for unmatched records
    missing_demo = result_df[result_df['Sex(M/F)'].isna()]
    if len(missing_demo) > 0:
        print(f"\nWarning: {len(missing_demo)} records failed to match demographic information")
        print("Unmatched Names:", missing_demo['Name'].tolist())
    else:
        print("\nAll records successfully matched with demographic information")
