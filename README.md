# Stock Market Data Processing & Analysis

This project processes raw stock market data, enhances it with additional indicators and pivot-based calculations, and outputs a more comprehensive dataset ready for further analysis. It also provides a PyQt5-based GUI for selecting raw CSV files, processing them, and visualizing final outputs and plots. 

> **Note**: This is **not** a production-level project—designed for practical experimentation and demonstration purposes.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Algorithm Complexity](#algorithm-complexity)
4. [Project Structure](#project-structure)
5. [Input Format](#input-format)
6. [Output Format](#output-format)
7. [Implementation Details](#implementation-details)
   1. [UI (PyQt5)](#ui-pyqt5)
   2. [Core Classes](#core-classes)
   3. [Concurrency](#concurrency)
8. [Example Visuals](#example-visuals)
9. [Next Steps](#next-steps)

---

## Overview

- **Goal**: Take raw CSV stock market data and produce enriched CSV output containing pivot points, support/resistance lines, trendlines, and technical indicators (EMA, RSI, MACD, Bollinger Bands, etc.).
- **Implementation**: Multi-core Python code (with a custom C++ extension for performance-critical parts), supplemented by a PyQt5 GUI.
- **Use Case**: Ideal for learning about advanced data manipulation and technical indicator calculations; not suited for high-frequency trading or mission-critical production.

---

## Key Features

- **Complex Pivot Detection**: Each row can reference up to 300 previous rows, leading to a factorial (_n!_) computational complexity for maximum accuracy and minimal data leakage.
- **Multi-Core Processing**: By default, the code utilizes multiple CPU cores for faster processing. You can tweak this by modifying the number of workers in the `ProcessPoolExecutor`.
- **Data Enhancement**: Adds pivot highs/lows, support/resistance lines, moving averages, RSI, MACD, Bollinger Bands, and more.
- **Interactive GUI**: A PyQt5 interface (`src/ui.py`) for selecting raw CSV files, applying parameters (start/end dates, pivot window, etc.), and processing them in parallel.
- **Trend Line Visualization**: Automatically or manually generate minimal candle charts with pivot-based trendlines. Final CSVs can be plotted to reveal detected pivots and extended lines.

---

## Algorithm Complexity

> **Complexity**: *O(n!)*

The factorial complexity arises because the algorithm performs similar calculations for each candle to ensure data integrity and prevent data leakage. Specifically, for every candle (or row), the system processes up to 300 previous candles to determine pivot points, support/resistance lines, and trendlines. This approach ensures that each calculation is based solely on historical data relative to the current candle, avoiding any inadvertent inclusion of future data.

For instance, identifying whether a candle is a pivot point requires evaluating its relationship with numerous preceding candles. Even if subsequent candles indicate that a previous candle was not a pivot, the information must still be recorded to maintain consistency in the dataset used for model training. This necessitates reprocessing each candle individually, resulting in operations that scale factorially with the number of candles.

This design prioritizes accuracy and data integrity, essential for training models that rely on time series data without future knowledge.

---

## Project Structure

Below is a simplified tree of the repository (omitting virtual environment files, build artifacts, etc.):

	.
	├── README.md
	├── data_manipulation
	│   ├── process_all_data.py
	│   ├── process_data.ipynb
	│   ├── raw_data
	│   └── train.py
	├── debug.ipynb
	├── images
	│   ├── Ekran Resmi 2025-01-18 16.11.22.png
	│   ├── Ekran Resmi 2025-01-18 16.12.03.png
	│   └── Ekran Resmi 2025-01-18 16.12.32.png
	├── pivots_s_r_lines.ipynb
	├── requirements.txt
	├── src
	│   ├── build
	│   ├── classes.py
	│   ├── collected_data
	│   ├── cpp_extensions
	│   │   ├── build
	│   │   ├── dist
	│   │   ├── mistaken_logic.cpp
	│   │   ├── mistaken_logic_ext.egg-info
	│   │   └── setup.py
	│   ├── data
	│   ├── indicators.py
	│   ├── main.py
	│   ├── minimal_candle_trendline_plot.py
	│   ├── output
	│   │   ├── output_csv
	│   │   └── output_plots
	│   ├── pivots.py
	│   ├── plotting.py
	│   ├── separate_data
	│   ├── sr_levels.py
	│   ├── synthetic.py
	│   ├── test.py
	│   ├── trendlines.py
	│   └── ui.py
	└── test
	└── scrape.py

---

## Input Format

Raw CSV files must contain the following columns (with optional timezone info):

	time,open,high,low,close,volume
	2024-01-11 09:00:00+0200,5.50,5.57,5.48,5.56,945999
	2024-01-11 10:00:00+0200,5.56,5.63,5.47,5.62,612429
	2024-01-11 11:00:00+0200,5.62,5.65,5.60,5.63,523800

•	time: Datetime in YYYY-MM-DD HH:MM:SS±HH:MM format
•	open, high, low, close: Standard OHLC data
•	volume: Trading volume for that period
•	Time intervals can be 1 hour (1H), daily, 15 minutes, or any other consistent frequency.

Example Input File: [`src/data/BIST_DLY_SYNTHETIC/60_aaaaa.csv`](src/data/BIST_DLY_SYNTHETIC/60_aaaaa.csv)

	time,open,high,low,close,volume
	2024-01-11 09:00:00+0200,5.50,5.57,5.48,5.56,945999
	2024-01-11 10:00:00+0200,5.56,5.63,5.47,5.62,612429
	2024-01-11 11:00:00+0200,5.62,5.65,5.60,5.63,523800

---

## Output Format

Enriched CSV output includes all original columns plus additional indicators, pivots, and more:

	time,open,high,low,close,volume,Pivot Highs,Pivot Lows,sr_lines,EMA20,EMA50,RSI14,MACD
	2024-01-13 01:00:00+00:00,5.38,5.46,5.37,5.45,332301,,5.31;5.36;5.41;5.48,5.50,5.55,70.5,0.25
	2024-01-13 02:00:00+00:00,5.45,5.50,5.40,5.48,289450,5.46,,5.35;5.40;5.45,5.52,5.60,68.3,0.20
	2024-01-13 03:00:00+00:00,5.48,5.55,5.44,5.54,310120,,5.36;5.42;5.47,5.58,5.65,72.1,0.30

Where additional columns may include:
	•	Pivot Highs / Pivot Lows
	•	Support/Resistance lines (sr_lines, sr_lines_weights)
	•	Moving Averages: EMA20, EMA50, EMA100, EMA200
	•	RSI (14, 7, etc.) and derived metrics
	•	MACD (signal & histogram)
	•	Bollinger Bands (upper, middle, lower)

Example Output File: [`src/output/output_csv/BIST_DLY_SYNTHETIC/60_aaaaa_final.csv`](src/output/output_csv/BIST_DLY_SYNTHETIC/60_aaaaa_final.csv)

Example Output Plot: [`src/output/output_plots/BIST_DLY_SYNTHETIC/60_aaaaa_chart.png`](src/output/output_plots/BIST_DLY_SYNTHETIC/60_aaaaa_chart.png)

---

## Implementation Details

### UI (PyQt5)

All UI-related code is in [`src/ui.py`](src/ui.py). This module creates a `MainWindow` class that:

1. **Lists Raw CSV Files**: In `data/`, scanning for `*.csv`.  
2. **Parameter Inputs**: Allows setting start date, end date, pivot window, etc.  
3. **Process Selected Files**: Runs the `process_stock()` function (from [`main.py`](src/main.py)) in **parallel** using a `ProcessPoolExecutor`.  
4. **Auto-Generating and Listing Plots**: Displays `.png` plot files in `output_plots` directory and opens them via a custom `PlotViewer` dialog.  
5. **Handling Final CSVs**: Lists files like `*_final.csv` in `output_csv`, then applies a `plot_trendlines_for_final_csv()` function for an optional minimal candle & trendline visualization.  

Key UI features:

- **Concurrent Processing**: By default, uses `max_workers=2` for the process pool.  
- **Automatic Column Normalization**: Renames any `'Volume'` column to `'volume'` for consistency.  
- **Interval & Ticker Detection**: Extracts the interval (1H, 4H, 15m, etc.) and ticker name from filenames.  
- **Selectable Lists**: Users can select/deselect multiple raw or final CSV files.  

### Core Classes

[`src/classes.py`](src/classes.py) defines fundamental data structures:

- **`PivotPoint`**: Represents a single pivot (high or low), storing date, OHLC, volume, and pivot type.  
- **`TrendLine`**: Tracks an initial pivot point (`starting_pivot`) and subsequent “included” pivots.  
  - **`calculate_end_point`** estimates a projected price/time for the trend line by extrapolating pivot movement.  
  - Allows additional pivot points (`add_pivot`) to be appended for a more complete trend model.  

These classes integrate with pivot detection (in scripts like [`pivots.py`](src/pivots.py)) and trendline logic (`trendlines.py`) to build a robust pivot/trend analysis system.

### Concurrency

- Uses `concurrent.futures.ProcessPoolExecutor` to parallelize `process_stock()`, so multiple CSV files can be processed simultaneously.  
- Each process runs the heavy logic (factorial complexity pivot detection, indicator calculations) independently.  
- After processing, the UI triggers a second pass to create or update final CSV files and generate trendline plots automatically.

---

## Example Visuals

Below are a few sample screenshots demonstrating the GUI, the final charts with pivot points, and minimal candle/trendline plots:

### 1. GUI Overview
![GUI Overview](images/Ekran%20Resmi%202025-01-18%2016.11.22.png)

### 2. Detailed Candlestick Chart
![Detailed Candlestick Chart](images/Ekran%20Resmi%202025-01-18%2016.12.03.png)

### 3. Minimal Candle & Trend Lines Plot
![Minimal Candle & Trend Lines Plot](images/Ekran%20Resmi%202025-01-18%2016.12.32.png)

---
