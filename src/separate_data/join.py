import os
import pandas as pd

# Directory containing the CSV files
directory = './'  # Adjust the directory path if needed

# Start date for filtering, making it timezone-aware
start_date = pd.Timestamp('2010-01-10T00:00:00+03:00')  # Match the timezone in your data (+03:00)

# List to store individual dataframes
dataframes = []

# Loop through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        # Read the CSV file, explicitly parsing the 'time' column
        df = pd.read_csv(filepath, parse_dates=['time'])
        
        # Filter rows where 'time' is greater than or equal to the start_date
        df = df[df['time'] >= start_date]
        dataframes.append(df)

# Combine all dataframes into one, removing duplicate rows based on the 'time' column
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=['time'])

# Sort by time column to maintain chronological order
combined_df = combined_df.sort_values(by='time')

# Write the combined dataframe to a single CSV file
output_filepath = os.path.join(directory, 'BIST_DLY_ARCLK, 60_aaaaa.csv')
combined_df.to_csv(output_filepath, index=False)

print(f"Filtered combined CSV file saved as: {output_filepath}")