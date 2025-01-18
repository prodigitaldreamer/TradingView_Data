import pandas as pd
import os
import re  # Import the regular expression module

def drop_rows_by_time(directory, time_pattern):
    """
    Drops rows from CSV files in a directory where the 'time' column
    contains a given pattern.

    Args:
        directory (str): The path to the directory containing CSV files.
        time_pattern (str): The regular expression pattern to search for in the 'time' column.
                            For example:
                             - "18:00:00" to match exactly "18:00:00".
                             - "18:..:.." to match the hour 18 and any value for minutes/seconds
                             - "18" to match any time that has 18 in its hours field

    Returns:
        None. Modifies the CSV files in place.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                if 'time' not in df.columns:
                    print(f"Skipping {filename}: 'time' column not found.")
                    continue
                
                # Use regex to find matching rows
                rows_to_drop = df['time'].apply(lambda x: bool(re.search(time_pattern, str(x))))
                
                df = df[~rows_to_drop] # Drop rows that matched the regex pattern
                
                df.to_csv(filepath, index=False)
                print(f"Processed {filename}: Rows matching '{time_pattern}' dropped.")

            except pd.errors.ParserError:
                print(f"Skipping {filename}: Could not parse CSV file.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    current_directory = "."  # Current directory
    time_to_remove = r"18:00:00" # The time part you want to filter out (as a regular expression)

    drop_rows_by_time(current_directory, time_to_remove)