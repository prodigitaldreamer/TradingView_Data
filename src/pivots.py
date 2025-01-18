# pivots.py

import numpy as np
import pandas as pd

def detect_pivots(filtered_data, pivot_window=5):
    if 'Pivot Highs' not in filtered_data.columns:
        filtered_data['Pivot Highs'] = np.nan
    if 'Pivot Lows' not in filtered_data.columns:
        filtered_data['Pivot Lows'] = np.nan

    # If not enough data, return immediately.
    if len(filtered_data) < pivot_window:
        return

    for i in range(pivot_window, len(filtered_data)):
        window = filtered_data.iloc[i - pivot_window:i]

        if len(window) == 0 or window['high'].dropna().empty or window['low'].dropna().empty:
            date = filtered_data.index[i]
            filtered_data.at[date, 'Pivot Highs'] = np.nan
            filtered_data.at[date, 'Pivot Lows'] = np.nan
            continue

        current_high = filtered_data['high'].iloc[i]
        current_low = filtered_data['low'].iloc[i]

        if pd.isna(current_high) or pd.isna(current_low):
            date = filtered_data.index[i]
            filtered_data.at[date, 'Pivot Highs'] = np.nan
            filtered_data.at[date, 'Pivot Lows'] = np.nan
            continue

        # Safely compute max/min
        window_high_values = window['high'].dropna()
        window_low_values = window['low'].dropna()

        if window_high_values.empty or window_low_values.empty:
            # No valid data
            date = filtered_data.index[i]
            filtered_data.at[date, 'Pivot Highs'] = np.nan
            filtered_data.at[date, 'Pivot Lows'] = np.nan
            continue

        window_high_max = window_high_values.max()
        window_low_min = window_low_values.min()

        if pd.isna(window_high_max) or pd.isna(window_low_min):
            date = filtered_data.index[i]
            filtered_data.at[date, 'Pivot Highs'] = np.nan
            filtered_data.at[date, 'Pivot Lows'] = np.nan
            continue

        is_pivot_high = current_high > window_high_max
        is_pivot_low = current_low < window_low_min

        date = filtered_data.index[i]
        filtered_data.at[date, 'Pivot Highs'] = current_high if is_pivot_high else np.nan
        filtered_data.at[date, 'Pivot Lows'] = current_low if is_pivot_low else np.nan