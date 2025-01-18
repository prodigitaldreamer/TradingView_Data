# plotting.py

import mplfinance as mpf
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

def plot_candlestick_with_pivots_and_sr_levels(filtered_data, sr_levels, time_interval):
    if len(filtered_data) == 0:
        print("No data to plot.")
        return None, None

    last_row = filtered_data.iloc[-1]
    vps_str = last_row['Valid Pivots']
    final_pivots = []
    if pd.notna(vps_str) and vps_str.strip():
        vps_list = vps_str.split(';')
        for vp in vps_list:
            parts = vp.split('@')
            if len(parts) < 3:
                print(f"Invalid pivot format '{vp}'. Skipping.")
                continue
            ptype = parts[0]
            vdate_str = parts[1]
            val_str = parts[2]
            # weight may exist but we do not need it for plotting pivots, ignore extra parts.

            try:
                vdate = pd.to_datetime(vdate_str)
                val = float(val_str)
                final_pivots.append((ptype, vdate, val))
            except Exception as e:
                print(f"Error parsing pivot '{vp}': {e}")

    high_vals = pd.Series(np.nan, index=filtered_data.index)
    low_vals = pd.Series(np.nan, index=filtered_data.index)

    for ptype, vdate, val in final_pivots:
        if vdate in filtered_data.index:
            if ptype == 'high':
                high_vals[vdate] = val
            else:
                low_vals[vdate] = val
        else:
            print(f"Pivot date {vdate} not found in filtered_data index, skipping pivot.")

    add_lines = []
    for lvl in sr_levels:
        add_lines.append(
            mpf.make_addplot([lvl]*len(filtered_data), color='purple', linestyle='--')
        )

    pivot_highs_plot = mpf.make_addplot(
        high_vals, type='scatter', markersize=100, marker='^', color='green', label='Pivot Highs'
    )
    pivot_lows_plot = mpf.make_addplot(
        low_vals, type='scatter', markersize=100, marker='v', color='red', label='Pivot Lows'
    )

    all_addplots = add_lines + [pivot_highs_plot, pivot_lows_plot]

    fig, axlist = mpf.plot(
        filtered_data,
        type='candle',
        style='charles',
        title='Candlestick Chart (Final)',
        ylabel='Price',
        volume=True,
        addplot=all_addplots,
        figsize=(14,10),
        tight_layout=True,
        returnfig=True,
        xrotation=15
    )

    ax = axlist[0]
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig.autofmt_xdate()

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    print("Plot completed with final pivots only.")
    return fig, axlist