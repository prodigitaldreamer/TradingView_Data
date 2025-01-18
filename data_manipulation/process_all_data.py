import os
import pandas as pd
import numpy as np

# If you want display() calls to work outside notebooks:
try:
    from IPython.display import display
except ImportError:
    # Define a no-op if IPython isn't available
    def display(*args, **kwargs):
        pass

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all the existing data manipulations that were done on df_sample
    (Pivot Highs/Lows, BB, EMA, SR lines, trendlines, RSI, etc.) and returns
    the transformed DataFrame.
    """
    print("\n# Initialize the NOTES dictionary")
    NOTES = {}

    print("\n# Process 'Pivot Highs' and 'Pivot Lows' columns")
    for col in ['Pivot Highs', 'Pivot Lows']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: 1 if pd.notna(x) else 0)
            NOTES['pivot values'] = 'data leak'  # Mark pivot values as data leak

    not_dealt = [col for col in df.columns
                 if col not in ['Pivot Highs', 'Pivot Lows']]
    print("Columns not yet dealt with:", not_dealt)

    print("\n# Calculate percentage difference for BB columns")
    for col in ['BB_upper', 'BB_middle', 'BB_lower']:
        if col in df.columns:
            df[col] = (df[col] - df['close']) / df['close']
            if col in not_dealt:
                not_dealt.remove(col)
            NOTES[col] = 'data leak'

    print("\n# Calculate percentage difference for EMA columns")
    for col in ['EMA20', 'EMA50', 'EMA100', 'EMA200']:
        if col in df.columns and col in not_dealt:
            df[col] = (df[col] - df['close']) / df['close']
            not_dealt.remove(col)
            NOTES[col] = 'data leak'

    print("\n# Create a new NOTES dictionary with desired structure")
    new_notes = {}
    new_notes['data leak'] = [key for key in NOTES]
    NOTES = new_notes
    print("Updated NOTES dictionary:", NOTES)

    print("Columns not yet dealt with:", not_dealt)
    NOTES['data leak'] += ['sr_lines', 'sr_lines_weights']

    print("\n# Extracting SR features")
    def extract_sr_features(row):
        close_price = row['close']
        sr_lines_str = row['sr_lines']
        sr_weights_str = row['sr_lines_weights']

        if pd.isna(sr_lines_str) or pd.isna(sr_weights_str):
            return [np.nan] * 13

        sr_lines = [float(x) for x in sr_lines_str.split(';')]
        sr_weights = [int(x) for x in sr_weights_str.split(';')]
        sr_pairs = list(zip(sr_lines, sr_weights))
        if not sr_pairs:
            return [np.nan] * 13

        below_lines = [(line, weight) for line, weight in sr_pairs
                       if line < close_price]
        above_lines = [(line, weight) for line, weight in sr_pairs
                       if line > close_price]
        highest_weight_lines = sorted(sr_pairs, key=lambda x: x[1], reverse=True)

        num_sr_below = len(below_lines)
        num_sr_above = len(above_lines)
        total_weight_below = sum(weight for _, weight in below_lines) if below_lines else 0
        total_weight_above = sum(weight for _, weight in above_lines) if above_lines else 0

        below_sr_features = [line for line, _ in
                             sorted(below_lines, key=lambda x: (close_price - x[0]))[:3]]
        above_sr_features = [line for line, _ in
                             sorted(above_lines, key=lambda x: (x[0] - close_price))[:3]]
        highest_weight_features = [line for line, _ in highest_weight_lines[:3]]

        while len(below_sr_features) < 3:
            below_sr_features.append(np.nan)
        while len(above_sr_features) < 3:
            above_sr_features.append(np.nan)
        while len(highest_weight_features) < 3:
            highest_weight_features.append(np.nan)

        return below_sr_features + above_sr_features + highest_weight_features + \
               [num_sr_below, num_sr_above, total_weight_below, total_weight_above]

    sr_features = df.apply(extract_sr_features, axis=1, result_type='expand')
    sr_features.columns = [
        'sr_below_1', 'sr_below_2', 'sr_below_3',
        'sr_above_1', 'sr_above_2', 'sr_above_3',
        'sr_weight_1', 'sr_weight_2', 'sr_weight_3',
        'num_sr_below', 'num_sr_above', 'total_weight_below', 'total_weight_above'
    ]
    df = pd.concat([df, sr_features], axis=1)
    if 'sr_lines' in not_dealt:
        not_dealt.remove('sr_lines')
    if 'sr_lines_weights' in not_dealt:
        not_dealt.remove('sr_lines_weights')

    print("Columns not yet dealt with:", not_dealt)
    print("Updated NOTES dictionary:", NOTES)

    print("\n# Drop 'sr_lines' and 'sr_lines_weights' from df")
    df = df.drop(columns=['sr_lines', 'sr_lines_weights'])
    print("Updated Columns:", df.columns.tolist())

    print("\n# Update sr_columns to store percentage difference from close")
    sr_columns = [
        'sr_below_1', 'sr_below_2', 'sr_below_3',
        'sr_above_1', 'sr_above_2', 'sr_above_3',
        'sr_weight_1', 'sr_weight_2', 'sr_weight_3'
    ]
    for col in sr_columns:
        if col in df.columns:
            df[col] = df.apply(
                lambda row: (row[col] - row['close']) / row['close']
                if pd.notna(row[col]) else row[col],
                axis=1
            )
    print("Support/Resistance columns updated to percentage difference values.")

    print("\n# Extract trendline features")
    def extract_trendline_features(row):
        close_price = row['close']
        trendlines_str = row['TrendLines']

        if pd.isna(trendlines_str):
            return [np.nan] * 13

        trendlines = []
        for tl in trendlines_str.split('|'):
            direction, price, weight = tl.split('@')
            trendlines.append((direction, float(price), int(weight)))

        below_lines = [(price, weight) for direction, price, weight in trendlines
                       if price < close_price]
        above_lines = [(price, weight) for direction, price, weight in trendlines
                       if price > close_price]

        below_lines.sort(key=lambda x: (close_price - x[0]))
        above_lines.sort(key=lambda x: (x[0] - close_price))

        all_lines = [(price, weight) for direction, price, weight in trendlines]
        highest_weight_lines = sorted(all_lines, key=lambda x: x[1], reverse=True)

        num_tl_below = len(below_lines)
        num_tl_above = len(above_lines)
        total_weight_below = sum(weight for _, weight in below_lines) if below_lines else 0
        total_weight_above = sum(weight for _, weight in above_lines) if above_lines else 0

        below_features = [(price - close_price) / close_price
                          for price, _ in below_lines[:3]]
        above_features = [(price - close_price) / close_price
                          for price, _ in above_lines[:3]]
        weight_features = [(price - close_price) / close_price
                           for price, _ in highest_weight_lines[:3]]

        while len(below_features) < 3:
            below_features.append(np.nan)
        while len(above_features) < 3:
            above_features.append(np.nan)
        while len(weight_features) < 3:
            weight_features.append(np.nan)

        return below_features + above_features + weight_features + [
            num_tl_below, num_tl_above, total_weight_below, total_weight_above
        ]

    tl_features = df.apply(extract_trendline_features, axis=1, result_type='expand')
    tl_features.columns = [
        'tl_below_1', 'tl_below_2', 'tl_below_3',
        'tl_above_1', 'tl_above_2', 'tl_above_3',
        'tl_weight_1', 'tl_weight_2', 'tl_weight_3',
        'num_tl_below', 'num_tl_above',
        'total_tl_weight_below', 'total_tl_weight_above'
    ]
    df = pd.concat([df, tl_features], axis=1)
    df = df.drop(columns=['Valid Pivots', 'TrendLines'])
    NOTES['data leak'] += ['Valid Pivots', 'TrendLines']

    print("\nColumns after processing:", df.columns.tolist())
    print("\nUpdated NOTES dictionary:", NOTES)

    print("\n# Process time column to extract features")
    def extract_time_features(time_str):
        dt = pd.to_datetime(time_str)
        dt = dt + pd.Timedelta(hours=2)
        month = dt.month
        day = dt.day
        day_of_week = dt.dayofweek
        hour = dt.hour
        if month in [3, 4, 5]:
            season = 0  # Spring
        elif month in [6, 7, 8]:
            season = 1  # Summer
        elif month in [9, 10, 11]:
            season = 2  # Autumn
        else:
            season = 3  # Winter
        return pd.Series({
            'month': month,
            'day': day,
            'day_of_week': day_of_week,
            'hour': hour,
            'season': season
        })

    if 'time' in df.columns:
        time_features = df['time'].apply(extract_time_features)
        df = pd.concat([df, time_features], axis=1)
        df = df.drop(columns=['time'])

    print("\nUpdated column list:")
    print(df.columns.tolist())

    print("\nUpdated NOTES dictionary:", NOTES)

    print("\n# Calculate price changes relative to previous close")
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            df[f'{col}_pct'] = df[col].pct_change()

    print("\n# Calculate rolling volume averages")
    if 'volume' in df.columns:
        volume_periods = [7, 21, 50, 100, 200]
        for period in volume_periods:
            df[f'volume_ratio_{period}'] = (
                df['volume'] /
                df['volume'].rolling(window=period).mean()
            )

    # Drop original price and volume columns if they exist
    existing_cols = [c for c in price_columns + ['volume'] if c in df.columns]
    df = df.drop(columns=existing_cols, errors='ignore')

    rename_dict = {
        'open_pct': 'open',
        'high_pct': 'high',
        'low_pct': 'low',
        'close_pct': 'close'
    }
    df = df.rename(columns=rename_dict)

    print("\nUpdated column list:")
    print(df.columns.tolist())

    print("\n# Create RSI-derived features")
    def calculate_rsi_features(df):
        df['rsi14_zone'] = pd.cut(
            df['RSI14'],
            bins=[-float('inf'), 30, 70, float('inf')],
            labels=[-1, 0, 1]
        )
        df['rsi7_zone'] = pd.cut(
            df['RSI7'],
            bins=[-float('inf'), 30, 70, float('inf')],
            labels=[-1, 0, 1]
        )
        df['rsi14_ma_div'] = (df['RSI14'] - df['RSI14_MA14']) / 100
        df['rsi7_ma_div'] = (df['RSI7'] - df['RSI7_MA7']) / 100
        df['rsi14_momentum'] = df['RSI14'].diff() / 100
        df['rsi7_momentum'] = df['RSI7'].diff() / 100

        df['rsi14_cross'] = np.where(
            (df['RSI14'].shift(1) < df['RSI14_MA14'].shift(1)) &
            (df['RSI14'] > df['RSI14_MA14']), 1,
            np.where(
                (df['RSI14'].shift(1) > df['RSI14_MA14'].shift(1)) &
                (df['RSI14'] < df['RSI14_MA14']), -1, 0
            )
        )

        df['rsi7_cross'] = np.where(
            (df['RSI7'].shift(1) < df['RSI7_MA7'].shift(1)) &
            (df['RSI7'] > df['RSI7_MA7']), 1,
            np.where(
                (df['RSI7'].shift(1) > df['RSI7_MA7'].shift(1)) &
                (df['RSI7'] < df['RSI7_MA7']), -1, 0
            )
        )
        df['rsi_agreement'] = (df['rsi14_zone'] == df['rsi7_zone']).astype(int)

        df = df.drop(columns=['RSI14', 'RSI7', 'RSI14_MA14', 'RSI7_MA7'], errors='ignore')
        return df

    df = calculate_rsi_features(df)

    print("\nFinal column list:")
    print(df.columns.tolist())

    return df

def main():
    print("# Data Preparation for Model Training")

    print("\n## Step 1: Confirm the raw_data structure")
    directory = 'raw_data/'
    output_folder = 'processed_data'
    os.makedirs(output_folder, exist_ok=True)

    csv_files = []
    columns_list = []

    for filename in os.listdir(directory):
        if filename.lower().endswith('.csv'):
            csv_files.append(os.path.join(directory, filename))

    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    for file in csv_files:
        try:
            # Only read the header row to check columns
            df_header = pd.read_csv(file, nrows=0)
            columns_list.append(set(df_header.columns))
        except Exception as e:
            print(f"Error reading {file}: {e}")

    first_columns = columns_list[0]
    all_same = all(cols == first_columns for cols in columns_list)
    if not all_same:
        print("Not all CSV files have the same columns.")
        differing_files = [
            csv_files[i] for i, cols in enumerate(columns_list)
            if cols != first_columns
        ]
        print("Files with differing columns:")
        for df_file in differing_files:
            print(f"- {df_file}")
        return

    print("\n# All CSV files share the same columns. Processing each file...")
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            processed_df = process_df(df)
            output_path = os.path.join(output_folder, os.path.basename(file))
            processed_df.to_csv(output_path, index=False)
            print(f"Saved processed data to: {output_path}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
