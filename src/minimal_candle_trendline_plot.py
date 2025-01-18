# minimal_candle_trendline_plot.py

import pyqtgraph as pg
import pandas as pd
import numpy as np
import math
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtCore import Qt

class MinimalCandleTrendlineWindow(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Minimal Candle & Trend Lines Plot")
        self.resize(1200, 600)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.data = data
        self.plot_data()

    def plot_data(self):
        data = self.data
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'TrendLines']):
            print("Data missing required columns.")
            return

        dates = data.index
        x_vals = np.arange(len(data))
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        for i in range(len(data)):
            x = x_vals[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]

            if c > o:
                color = (0,255,0)
            elif c < o:
                color = (255,0,0)
            else:
                color = (200,200,200)

            self.plot_widget.plot([x,x],[l,h], pen=pg.mkPen(color, width=1))
            self.plot_widget.plot([x-0.1, x+0.1],[o,o], pen=pg.mkPen(color, width=1))
            self.plot_widget.plot([x-0.1, x+0.1],[c,c], pen=pg.mkPen(color, width=1))

        trend_highs_x = []
        trend_highs_y = []
        trend_lows_x = []
        trend_lows_y = []

        for i, row in data.iterrows():
            trend_str = row['TrendLines']
            if pd.isna(trend_str) or trend_str.strip() == '':
                continue
            parts = trend_str.split('|')
            row_x = np.where(dates == i)[0]
            if len(row_x) == 0:
                continue
            xx = row_x[0]

            for part in parts:
                part = part.strip()
                segments = part.split('@')
                if len(segments) < 2:
                    continue
                line_type = segments[0]
                try:
                    val = float(segments[1])
                except:
                    continue
                # Ignore weight for plotting
                if line_type == 'H':
                    trend_highs_x.append(xx)
                    trend_highs_y.append(val)
                elif line_type == 'L':
                    trend_lows_x.append(xx)
                    trend_lows_y.append(val)

        if trend_highs_x:
            self.plot_widget.plot(trend_highs_x, trend_highs_y, pen=None, symbol='o', symbolBrush='b', symbolSize=5, name='High TL Points')
        if trend_lows_x:
            self.plot_widget.plot(trend_lows_x, trend_lows_y, pen=None, symbol='o', symbolBrush='orange', symbolSize=5, name='Low TL Points')

        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left','Price')
        self.plot_widget.setLabel('bottom','Index')
        self.plot_widget.showGrid(x=True,y=True)

        N = max(1, len(dates)//10)
        ticks = [(x_vals[i], dates[i].strftime('%Y-%m-%d')) for i in range(0,len(dates),N)]
        ax = self.plot_widget.getPlotItem().getAxis('bottom')
        ax.setTicks([ticks])

        print("Trend lines and minimal candles plotted using PyQtGraph.")