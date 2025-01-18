# trendlines.py

from classes import TrendLine
import pandas as pd
import matplotlib.dates as mdates

def find_trend_lines(filtered_data, pivot_points):
    """
    pivot_points: a list of PivotPoint objects created from the current sub_data.
    We'll form trend lines by connecting consecutive pivots of the same type:
    low pivots form uptrend lines, high pivots form downtrend lines.
    """
    from classes import TrendLine

    # separate into highs and lows
    pivot_highs = [(p, filtered_data.index.get_loc(p.date)) for p in pivot_points if p.pivot_type == 'high']
    pivot_lows = [(p, filtered_data.index.get_loc(p.date)) for p in pivot_points if p.pivot_type == 'low']

    trend_lines = []
    max_index = len(filtered_data)

    # Uptrend lines from lows
    for i in range(len(pivot_lows) - 1):
        start_pvt, start_idx = pivot_lows[i]
        end_pvt, end_idx = pivot_lows[i+1]
        t = TrendLine(start_pvt, start_idx)
        t.add_pivot(end_pvt, end_idx)
        t.calculate_end_point(max_index, filtered_data)
        trend_lines.append(t)

    # Downtrend lines from highs
    for i in range(len(pivot_highs) - 1):
        start_pvt, start_idx = pivot_highs[i]
        end_pvt, end_idx = pivot_highs[i+1]
        t = TrendLine(start_pvt, start_idx)
        t.add_pivot(end_pvt, end_idx)
        t.calculate_end_point(max_index, filtered_data)
        trend_lines.append(t)

    return trend_lines