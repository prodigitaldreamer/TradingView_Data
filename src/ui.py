# ui.py

import sys
import os
import glob
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QFormLayout, QMessageBox, QDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from main import process_stock, plot_trendlines_for_final_csv
from minimal_candle_trendline_plot import MinimalCandleTrendlineWindow

import concurrent.futures

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

def extract_ticker_from_filename(filename):
    parts = filename.split(",")
    if len(parts) < 2:
        return 'unknown'
    left_part = parts[0]
    if left_part.startswith("BIST_DLY_"):
        ticker = left_part[len("BIST_DLY_"):]
        return ticker
    return 'unknown'

class PlotViewer(QDialog):
    def __init__(self, plot_path):
        super().__init__()
        self.setWindowTitle("Plot Viewer")
        layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap(plot_path)
        if pixmap.isNull():
            label.setText("Could not load plot.")
        else:
            label.setPixmap(pixmap)
            label.setScaledContents(True)
        layout.addWidget(label)
        self.setLayout(layout)
        self.resize(800, 600)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Data Processor UI")

        # Directories inside src/
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        self.output_csv_dir = os.path.join(self.output_dir, 'output_csv')
        self.output_plots_dir = os.path.join(self.output_dir, 'output_plots')

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_csv_dir, exist_ok=True)
        os.makedirs(self.output_plots_dir, exist_ok=True)

        self.normalize_csv_columns()

        # Left panel: raw CSV files
        self.csv_list = QListWidget()
        self.csv_list.setSelectionMode(QListWidget.NoSelection)
        self.refresh_csv_button = QPushButton("Refresh Raw CSV List")
        self.select_all_button = QPushButton("Select All Raw")
        self.deselect_all_button = QPushButton("Deselect All Raw")

        self.refresh_csv_button.clicked.connect(self.load_csv_files)
        self.select_all_button.clicked.connect(self.select_all_csv)
        self.deselect_all_button.clicked.connect(self.deselect_all_csv)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Raw CSV Files:"))
        left_layout.addWidget(self.csv_list)
        left_layout.addWidget(self.refresh_csv_button)
        left_layout.addWidget(self.select_all_button)
        left_layout.addWidget(self.deselect_all_button)

        # Middle panel: Parameters and process
        self.start_date_input = QLineEdit("2024-09-01")
        self.end_date_input = QLineEdit("2024-12-01")
        self.pivot_window_input = QLineEdit("3")
        self.abs_eps_input = QLineEdit("0.5")
        self.min_samples_input = QLineEdit("1")

        form = QFormLayout()
        form.addRow("Start Date (YYYY-MM-DD):", self.start_date_input)
        form.addRow("End Date (YYYY-MM-DD):", self.end_date_input)
        form.addRow("Pivot Window:", self.pivot_window_input)
        form.addRow("abs_eps:", self.abs_eps_input)
        form.addRow("min_samples:", self.min_samples_input)

        self.process_button = QPushButton("Process Selected Raw Files")
        self.process_button.clicked.connect(self.process_selected_files)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel("Parameters:"))
        middle_layout.addLayout(form)
        middle_layout.addWidget(self.process_button)

        # Right panel: plots
        self.plot_list = QListWidget()
        self.refresh_plots_button = QPushButton("Refresh Plots List")
        self.open_plot_button = QPushButton("Open Selected Plot")

        self.refresh_plots_button.clicked.connect(self.load_plots)
        self.open_plot_button.clicked.connect(self.open_selected_plot)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Generated Plots:"))
        right_layout.addWidget(self.plot_list)
        right_layout.addWidget(self.refresh_plots_button)
        right_layout.addWidget(self.open_plot_button)

        # Final CSV list for trendline plotting
        self.final_csv_list = QListWidget()
        self.final_csv_list.setSelectionMode(QListWidget.NoSelection)
        self.refresh_final_button = QPushButton("Refresh Final CSV List")
        self.select_all_final_button = QPushButton("Select All Final")
        self.deselect_all_final_button = QPushButton("Deselect All Final")
        self.plot_trendlines_button = QPushButton("Plot Trend Lines Points")

        self.refresh_final_button.clicked.connect(self.load_final_csv_files)
        self.select_all_final_button.clicked.connect(self.select_all_final_csv)
        self.deselect_all_final_button.clicked.connect(self.deselect_all_final_csv)
        self.plot_trendlines_button.clicked.connect(self.plot_selected_final_csv_trendlines)

        final_layout = QVBoxLayout()
        final_layout.addWidget(QLabel("Final CSV Files:"))
        final_layout.addWidget(self.final_csv_list)
        final_layout.addWidget(self.refresh_final_button)
        final_layout.addWidget(self.select_all_final_button)
        final_layout.addWidget(self.deselect_all_final_button)
        final_layout.addWidget(self.plot_trendlines_button)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(right_layout)
        main_layout.addLayout(final_layout)

        self.setLayout(main_layout)

        self.load_csv_files()
        self.load_plots()
        self.load_final_csv_files()

    def normalize_csv_columns(self):
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'Volume' in df.columns:
                    df.rename(columns={'Volume': 'volume'}, inplace=True)
                    df.to_csv(csv_file, index=False)
                    print(f"Renamed 'Volume' to 'volume' in {csv_file}")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    def load_csv_files(self):
        self.csv_list.clear()
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        for f in csv_files:
            filename = os.path.basename(f)
            interval = detect_interval_from_filename(filename)
            ticker = extract_ticker_from_filename(filename)
            display_text = f"[{ticker}] {interval.upper()} : {filename}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.csv_list.addItem(item)

    def select_all_csv(self):
        for i in range(self.csv_list.count()):
            item = self.csv_list.item(i)
            item.setCheckState(Qt.Checked)

    def deselect_all_csv(self):
        for i in range(self.csv_list.count()):
            item = self.csv_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def load_plots(self):
        self.plot_list.clear()
        png_files = glob.glob(os.path.join(self.output_plots_dir, "*.png"))
        for f in png_files:
            fname = os.path.basename(f)
            item = QListWidgetItem(fname)
            item.setData(Qt.UserRole, f)
            self.plot_list.addItem(item)

    def process_selected_files(self):
        selected_files = []
        for i in range(self.csv_list.count()):
            item = self.csv_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_files.append(item.data(Qt.UserRole))

        if not selected_files:
            QMessageBox.warning(self, "No Files Selected", "Please select at least one CSV file.")
            return

        # Gather user inputs, etc...
        start_date = self.start_date_input.text().strip() or "2024-01-01"
        end_date   = self.end_date_input.text().strip()   or "2024-12-01"
        pivot_window = int(self.pivot_window_input.text().strip() or 5)
        abs_eps      = float(self.abs_eps_input.text().strip()    or 3)
        min_samples  = int(self.min_samples_input.text().strip()  or 1)

        # == Run in parallel ==
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for csv_file in selected_files:
                futures.append(
                    executor.submit(
                        process_stock,
                        csv_file,
                        start_date=start_date,
                        end_date=end_date,
                        pivot_window=pivot_window,
                        abs_eps=abs_eps,
                        min_samples=min_samples
                    )
                )
            concurrent.futures.wait(futures)

        QMessageBox.information(self, "Processing Complete", "Selected files have been processed.")
        self.load_plots()
        self.load_final_csv_files()

        # === AUTOMATICALLY run "plot_trendlines_for_final_csv" on each new final CSV ===
        for i in range(self.final_csv_list.count()):
            final_csv_name = self.final_csv_list.item(i).data(Qt.UserRole)
            final_csv_path = os.path.join(self.output_csv_dir, final_csv_name)
            if not os.path.exists(final_csv_path):
                continue

            # This calls the same function used by the "Plot Trend Lines Points" button
            data = plot_trendlines_for_final_csv(final_csv_path)
            if data is not None:
                # If you truly need to display the PyQt window, do something like:
                #   trend_window = MinimalCandleTrendlineWindow(data, parent=self)
                #   trend_window.exec_()
                #
                # But if you just need the final code to update the CSV or do any logic,
                # calling plot_trendlines_for_final_csv() is enough. 
                pass

        QMessageBox.information(self, "Trendlines Complete", 
                                "Automatic trendline plotting done for all final CSVs.")

    def load_final_csv_files(self):
        self.final_csv_list.clear()
        final_csv_files = glob.glob(os.path.join(self.output_csv_dir, "*_final.csv"))
        for f in final_csv_files:
            fname = os.path.basename(f)
            item = QListWidgetItem(fname)
            item.setData(Qt.UserRole, fname)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.final_csv_list.addItem(item)

    def select_all_final_csv(self):
        for i in range(self.final_csv_list.count()):
            item = self.final_csv_list.item(i)
            item.setCheckState(Qt.Checked)

    def deselect_all_final_csv(self):
        for i in range(self.final_csv_list.count()):
            item = self.final_csv_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def plot_selected_final_csv_trendlines(self):
        selected_final = []
        for i in range(self.final_csv_list.count()):
            item = self.final_csv_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_final.append(item.data(Qt.UserRole))

        if not selected_final:
            QMessageBox.warning(self, "No Final CSV Selected", "Please select at least one final CSV file.")
            return

        for final_csv in selected_final:
            filepath = os.path.join(self.output_csv_dir, final_csv)
            if not os.path.exists(filepath):
                QMessageBox.critical(self, "File Not Found", f"{final_csv} not found in output_csv directory.")
                continue
            data = plot_trendlines_for_final_csv(filepath)
            if data is not None:
                trend_window = MinimalCandleTrendlineWindow(data, parent=self)
                trend_window.exec_()

        QMessageBox.information(self, "Plotting Complete", "Trend lines points plotting done.")
        self.load_plots()

    def open_selected_plot(self):
        selected_items = self.plot_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Plot Selected", "Please select a plot from the list.")
            return
        plot_path = os.path.join(self.output_plots_dir, selected_items[0].data(Qt.UserRole))
        dlg = PlotViewer(plot_path)
        dlg.exec_()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1600, 800)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()