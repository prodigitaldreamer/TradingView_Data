# main.py

import os
import pandas as pd
import numpy as np
import math
from pivots import detect_pivots
from sr_levels import cluster_pivots
from plotting import plot_candlestick_with_pivots_and_sr_levels
from classes import PivotPoint
from indicators import add_indicators_to_csv

import mistaken_logic_ext

def detect_interval_from_filename(filename):
    if ", 1W_" in filename:
        return 'weekly'
    elif ", 1D_" in filename:
        return 'daily'
    elif ", 240_" in filename:
        return '4h'
    elif ", 60_" in filename:
        return '1h'
    elif ", 15_" in filename:
        return '15m'
    else:
        return 'unknown'
    
interval_scaling_map = {
    '1W': 1,      # Weekly
    '1D': 1,      # Daily
    '4H': 2,      # 4-Hour
    '1H': 8,      # Hourly
    '15M': 32,    # 15-Minute
    'unknown': 1, # Fallback
}


def apply_mistaken_logic(sub_data):
    # Ensure these columns exist
    if 'Pivot High Mistaken' not in sub_data.columns:
        sub_data['Pivot High Mistaken'] = np.nan
    if 'Pivot Low Mistaken' not in sub_data.columns:
        sub_data['Pivot Low Mistaken'] = np.nan

    # Convert 'Pivot Highs' and 'Pivot Lows' to float arrays (NaN for no pivot)
    pivot_high_arr = sub_data['Pivot Highs'].values.astype(float)
    pivot_low_arr  = sub_data['Pivot Lows'].values.astype(float)

    # Call the C++ function
    # It returns two boolean arrays indicating which rows are "mistaken"
    out_high, out_low = mistaken_logic_ext.mark_mistakes(pivot_high_arr, pivot_low_arr)

    # Convert those boolean arrays to Python-friendly True/False
    # and assign back to the DataFrame:
    sub_data['Pivot High Mistaken'] = out_high
    sub_data['Pivot Low Mistaken']  = out_low

    # The rest is your weighting logic that was originally in the second half:
    pivots_stack = []
    # But note: we've already done the "stack" logic in C++.
    # So if your "weighted_pivots_stack" relies on the final pivot stack,
    # you can skip the first half of your code. Just replicate the *weighting* portion:

    # Make a new list from the final stack you might want.
    # But if that stack is gone from Python, you might replicate the weighting in C++, too.

    # -- For demonstration, let's keep your weighting logic as is:
    # Now we read from the sub_data again. The "Pivot High Mistaken"/"Pivot Low Mistaken"
    # columns are updated. Then we do the weight calculations the same way we did before:
    #  (We'll rely on the final 'pivots_stack' that C++ ended with, etc.)

    # Maybe you do something like gather "remaining" pivots, then weigh them:
    remain_stack = []
    for idx, row in sub_data.iterrows():
        # If "Pivot Highs" is not NaN and "Pivot High Mistaken" is False => pivot is real
        if pd.notna(row['Pivot Highs']) and row['Pivot High Mistaken'] is False:
            remain_stack.append((idx, 'high', row['Pivot Highs']))
        if pd.notna(row['Pivot Lows']) and row['Pivot Low Mistaken'] is False:
            remain_stack.append((idx, 'low', row['Pivot Lows']))

    # After remain_stack is built, apply weighting to remain_stack:
    weighted_pivots_stack = []
    for i, (p_idx_val, ptype, pval) in enumerate(remain_stack):
        current_idx_num = sub_data.index.get_loc(p_idx_val)
        # find last opposite pivot in remain_stack
        opposite_type = 'low' if ptype == 'high' else 'high'
        last_opposite_idx_num = None
        for j in range(i-1, -1, -1):
            if remain_stack[j][1] == opposite_type:
                last_opposite_idx_val = remain_stack[j][0]
                last_opposite_idx_num = sub_data.index.get_loc(last_opposite_idx_val)
                break

        if last_opposite_idx_num is not None:
            candle_count = current_idx_num - last_opposite_idx_num
            if candle_count <= 7:
                base_weight = 1
            elif candle_count <= 14:
                base_weight = 2
            else:
                base_weight = 3
        else:
            base_weight = 1

        # Add previous same-type pivot weight -1
        prev_same_weight = None
        for k in range(i-1, -1, -1):
            if remain_stack[k][1] == ptype:
                prev_same_weight = weighted_pivots_stack[k][3]
                break

        if prev_same_weight is not None and prev_same_weight >= 3:
            final_weight = base_weight + (prev_same_weight - 2)
        else:
            final_weight = base_weight

        weighted_pivots_stack.append((p_idx_val, ptype, pval, final_weight))

    valid_pivots_list = []
    for (idx_val, ptype, val, w) in weighted_pivots_stack:
        valid_pivots_list.append(f"{ptype}@{idx_val.isoformat()}@{val}@{w}")

    return valid_pivots_list

def process_stock(csv_filename, start_date='2024-01-01', end_date='2024-12-01',
                  pivot_window=5, abs_eps=3, min_samples=1):
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    output_csv_dir = os.path.join(output_dir, 'output_csv')
    output_plots_dir = os.path.join(output_dir, 'output_plots')

    csv_file = os.path.join(data_dir, csv_filename)
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' does not exist.")
        return

    raw_data = pd.read_csv(csv_file)
    raw_data['time'] = pd.to_datetime(raw_data['time'], utc=True, errors='coerce')
    raw_data = raw_data.dropna(subset=['time'])
    raw_data.set_index('time', inplace=True)
    raw_data = raw_data[(raw_data.index >= start_date) & (raw_data.index <= end_date)].copy()
    raw_data = raw_data.sort_index()

    data = raw_data.copy()
    if 'Pivot Highs' not in data.columns:
        data['Pivot Highs'] = np.nan
    if 'Pivot Lows' not in data.columns:
        data['Pivot Lows'] = np.nan

    detect_pivots(data, pivot_window=pivot_window)

    if 'sr_lines' not in data.columns:
        data['sr_lines'] = ''
    if 'sr_lines_weights' not in data.columns:
        data['sr_lines_weights'] = ''
    if 'Pivot High Mistaken' not in data.columns:
        data['Pivot High Mistaken'] = False
    if 'Pivot Low Mistaken' not in data.columns:
        data['Pivot Low Mistaken'] = False
    if 'Valid Pivots' not in data.columns:
        data['Valid Pivots'] = ''
    if 'TrendLines' not in data.columns:
        data['TrendLines'] = ''

    TOL = 1e-8
    max_pivot_age = 300

    col_pivot_high_mistaken = data.columns.get_loc('Pivot High Mistaken')
    col_pivot_low_mistaken = data.columns.get_loc('Pivot Low Mistaken')
    col_valid_pivots = data.columns.get_loc('Valid Pivots')
    col_trendlines = data.columns.get_loc('TrendLines')
    col_sr_lines = data.columns.get_loc('sr_lines')
    col_sr_lines_weights = data.columns.get_loc('sr_lines_weights')

    for i in range(len(data)):
        sub_data = data.iloc[:i+1].copy()

        valid_pivots_list = apply_mistaken_logic(sub_data)

        # Filter out old pivots by weight
        recent_pivots_list = []
        for vp in valid_pivots_list:
            parts = vp.split('@')
            if len(parts) != 4:
                continue
            ptype, vdate_str, val_str, w_str = parts
            try:
                vdate = pd.to_datetime(vdate_str)
                val = float(val_str)
                w = float(w_str)
            except:
                continue
            if vdate in data.index:
                p_idx = data.index.get_loc(vdate)
                if (i - p_idx) <= max_pivot_age:
                    recent_pivots_list.append(vp)

        valid_pivots_list = recent_pivots_list

        if valid_pivots_list:
            data.iloc[i, col_valid_pivots] = ';'.join(valid_pivots_list)
        else:
            data.iloc[i, col_valid_pivots] = ''

        data.loc[sub_data.index, 'Pivot High Mistaken'] = sub_data['Pivot High Mistaken']
        data.loc[sub_data.index, 'Pivot Low Mistaken'] = sub_data['Pivot Low Mistaken']

        # Separate pivot highs/lows and their weights
        highs = []
        highs_w = []
        lows = []
        lows_w = []

        for vp in valid_pivots_list:
            ptype, vdate_str, val_str, w_str = vp.split('@')
            val = float(val_str)
            w = float(w_str)
            if ptype == 'high':
                highs.append(val)
                highs_w.append(w)
            else:
                lows.append(val)
                lows_w.append(w)

        pivot_high_values = np.array(highs) if highs else np.array([])
        pivot_high_weights = np.array(highs_w) if highs_w else np.array([])
        pivot_low_values = np.array(lows) if lows else np.array([])
        pivot_low_weights = np.array(lows_w) if lows_w else np.array([])

        # s/r lines weighted clusters
        if len(pivot_high_values) == 0 and len(pivot_low_values) == 0:
            data.iloc[i, col_sr_lines] = ''
            data.iloc[i, col_sr_lines_weights] = ''
        else:
            sr_levels, sr_counts = cluster_pivots(pivot_high_values, pivot_high_weights, 
                                                  pivot_low_values, pivot_low_weights,
                                                  abs_eps=abs_eps, min_samples=min_samples)
            if sr_levels:
                sr_str = ';'.join(str(round(val, 2)) for val in sr_levels)
                sr_wstr = ';'.join(str(int(count)) for count in sr_counts)
                data.iloc[i, col_sr_lines] = sr_str
                data.iloc[i, col_sr_lines_weights] = sr_wstr
            else:
                data.iloc[i, col_sr_lines] = ''
                data.iloc[i, col_sr_lines_weights] = ''

        # Trendlines with pivot weights
        vps = data.iloc[i]['Valid Pivots']
        if pd.isna(vps) or vps.strip() == '':
            data.iloc[i, col_trendlines] = ''
            continue

        vps_list = vps.split(';') if vps else []

        valid_highs = []
        valid_lows = []
        for vp in vps_list:
            parts = vp.split('@')
            if len(parts) < 3:
                print(f"Invalid pivot format '{vp}' at row {i}. Skipping.")
                continue
            ptype = parts[0]
            vdate_str = parts[1]
            val_str = parts[2]
            w_str = parts[3] if len(parts) >= 4 else '1'  # default weight if not present

            try:
                vdate = pd.to_datetime(vdate_str)
                val = float(val_str)
                w = float(w_str)
            except Exception as e:
                print(f"Error parsing pivot '{vp}' at row {i}: {e}")
                continue

            if vdate not in data.iloc[:i+1].index:
                print(f"Pivot date '{vdate}' not found in sub_data index at row {i}.")
                continue
            p_idx = data.iloc[:i+1].index.get_loc(vdate)

            if ptype == 'high':
                valid_highs.append((p_idx, vdate, val, w))
            elif ptype == 'low':
                valid_lows.append((p_idx, vdate, val, w))

        all_pivots_for_conditions = valid_highs + valid_lows

        def pivots_on_line(pivots_array, ptype_label, i, all_pivots_for_conditions):
            lines_prices = []
            n = len(pivots_array)
            for a in range(n):
                for b in range(a+1, n):
                    p0_idx, p0_date, p0_val, p0_w = pivots_array[a]
                    p1_idx, p1_date, p1_val, p1_w = pivots_array[b]

                    # 1) Decide how many days each candle covers for each interval
                    #    (You can refine or tweak these values as needed)
                    # Add this mapping at the top of main.py or within the process_stock function

                    # Suppose we get the interval name from the filename or from UI:
                    interval_name = detect_interval_from_filename(csv_file)  # e.g. "1W", "1D", "15M", etc.
                    scaling_factor = interval_scaling_map.get(interval_name, 1.0)  # Changed from days_per_candle

                    base_limit = 0.1  # 0.5% per day
                    limit_per_candle = base_limit / scaling_factor  # NEW: Adjusted limit per candle based on interval

                    # 2) Now the loop that checks slope:
                    dx = p1_idx - p0_idx
                    if dx == 0:
                        continue

                    if p0_val != 0:
                        # "per_candle_pct_change"
                        raw_slope = ((p1_val - p0_val) / p0_val) * 100.0 / dx
                        # Compare to the new limit
                        if abs(raw_slope) > limit_per_candle:  # CHANGED: Use limit_per_candle instead of daily_pct_change
                            # Skip line as "too steep"
                            continue

                    dy = p1_val - p0_val
                    slope = dy / dx  # Define slope based on price difference and index difference
                    line_pivots = [(p0_idx, p0_date, p0_val, p0_w), (p1_idx, p1_date, p1_val, p1_w)]
                    for c in range(n):
                        if c == a or c == b:
                            continue
                        pc_idx, pc_date, pc_val, pc_w = pivots_array[c]
                        expected_val = p0_val + slope * (pc_idx - p0_idx)
                        if math.isclose(pc_val, expected_val, abs_tol=TOL):
                            line_pivots.append((pc_idx, pc_date, pc_val, pc_w))

                    line_pivots.sort(key=lambda x: x[0])
                    if len(line_pivots) < 2:
                        continue
                    first_idx = line_pivots[0][0]
                    last_idx = line_pivots[-1][0]

                    dx_line = last_idx - first_idx
                    if dx_line == 0:
                        continue
                    dy_line = line_pivots[-1][2] - line_pivots[0][2]
                    final_slope = dy_line / dx_line
                    max_extension_idx = last_idx + dx_line * 2

                    if i > max_extension_idx:
                        continue

                    extension = i - first_idx
                    line_price = line_pivots[0][2] + final_slope * extension
                    line_type = 'H' if ptype_label == 'high' else 'L'

                    # Check conditions (already existing logic)
                    first_condition_met = False
                    second_condition_met = False
                    for pvt in all_pivots_for_conditions:
                        pc_idx2, pc_date2, pc_val2, pc_w2 = pvt
                        if pc_idx2 < first_idx or pc_idx2 > i:
                            continue
                        expected_val2 = line_pivots[0][2] + final_slope*(pc_idx2 - first_idx)
                        if line_type == 'H':
                            if not first_condition_met:
                                if pc_val2 > expected_val2:
                                    first_condition_met = True
                            else:
                                if pc_val2 < expected_val2:
                                    second_condition_met = True
                                    break
                        else: # L line
                            if not first_condition_met:
                                if pc_val2 < expected_val2:
                                    first_condition_met = True
                            else:
                                if pc_val2 > expected_val2:
                                    second_condition_met = True
                                    break

                    if first_condition_met and second_condition_met:
                        continue

                    # Trendline weight calculation (existing logic)
                    line_weight = sum(lp[3] for lp in line_pivots)
                    lines_prices.append(f"{line_type}@{round(line_price,2)}@{int(line_weight)}")

            return lines_prices

        high_lines = pivots_on_line(valid_highs, 'high', i, all_pivots_for_conditions) if valid_highs else []
        low_lines = pivots_on_line(valid_lows, 'low', i, all_pivots_for_conditions) if valid_lows else []
        all_lines = high_lines + low_lines

        if all_lines:
            data.iloc[i, data.columns.get_loc('TrendLines')] = '|'.join(all_lines)
        else:
            data.iloc[i, data.columns.get_loc('TrendLines')] = ''

    final_csv_path = os.path.join(output_csv_dir, f"{os.path.splitext(csv_filename)[0]}_final.csv")
    data.to_csv(final_csv_path)
    print("Processing complete. Final data saved to:", final_csv_path)

    plot_final_chart(data, os.path.splitext(csv_filename)[0])
    add_indicators_to_csv(final_csv_path)

def plot_final_chart(data, base_filename):
    last_row = data.iloc[-1]
    sr_lines_str = last_row['sr_lines'] if pd.notna(last_row['sr_lines']) else ''
    trend_lines_str = last_row['TrendLines'] if pd.notna(last_row['TrendLines']) else ''

    sr_levels = []
    if sr_lines_str.strip():
        sr_levels = [float(x) for x in sr_lines_str.split(';')]

    fig, axlist = plot_candlestick_with_pivots_and_sr_levels(data, sr_levels, 'custom')
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'output')
    output_plots_dir = os.path.join(output_dir, 'output_plots')
    os.makedirs(output_plots_dir, exist_ok=True)
    plot_path = os.path.join(output_plots_dir, f"{base_filename}_chart.png")
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def plot_trendlines_for_final_csv(final_csv_path):
    if not os.path.exists(final_csv_path):
        print(f"Final CSV '{final_csv_path}' does not exist.")
        return None
    data = pd.read_csv(final_csv_path, parse_dates=['time'], index_col='time')
    print(f"Data loaded for plotting trend lines from {final_csv_path}")
    return data