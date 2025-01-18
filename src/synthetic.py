import csv
import random
from datetime import datetime, timedelta, timezone
import os

def generate_synthetic_data(start_date, num_rows, output_path):
    """
    Generates synthetic stock market data and writes it to a CSV file.

    :param start_date: The starting datetime for the data.
    :param num_rows: Number of rows of data to generate.
    :param output_path: Path to save the generated CSV file.
    """
    headers = ['time', 'open', 'high', 'low', 'close', 'volume']
    data = []

    # Define timezone offset (+02:00)
    tz = timezone(timedelta(hours=2))
    current_time = start_date.replace(tzinfo=tz)

    # Initialize the first open price
    price = round(random.uniform(5.0, 10.0), 2)

    for _ in range(num_rows):
        # Simulate price changes
        open_price = price
        high_price = round(open_price + random.uniform(0.0, 0.1), 2)
        low_price = round(open_price - random.uniform(0.0, 0.1), 2)
        close_price = round(random.uniform(low_price, high_price), 2)
        volume = random.randint(300, 1000000)

        # Append the row
        data.append([
            current_time.strftime('%Y-%m-%d %H:%M:%S%z'),
            open_price,
            high_price,
            low_price,
            close_price,
            volume
        ])

        # Prepare for next row
        price = close_price
        current_time += timedelta(hours=1)

    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"Synthetic data generated and saved to {output_path}")

if __name__ == "__main__":
    # Define parameters
    START_DATE = datetime(2024, 1, 11, 9, 0, 0)  # Starting from 2010-01-11 09:00:00
    NUM_ROWS = 300  # Number of rows to generate
    OUTPUT_DIR = 'src/data'
    OUTPUT_FILENAME = 'BIST_DLY_SYNTHETIC, 60_aaaaa.csv'  # Updated filename
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate synthetic data
    generate_synthetic_data(START_DATE, NUM_ROWS, OUTPUT_PATH)