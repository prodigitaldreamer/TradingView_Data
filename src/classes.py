# classes.py

class PivotPoint:
    def __init__(self, date, open_, high, low, close, volume, pivot_type):
        self.date = date
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.pivot_type = pivot_type  # 'high' or 'low'
        # No mistaken logic needed.

    def __repr__(self):
        return (f"PivotPoint(date={self.date}, type={self.pivot_type}, O={self.open}, H={self.high}, "
                f"L={self.low}, C={self.close}, V={self.volume})")


class TrendLine:
    def __init__(self, starting_pivot, starting_index):
        self.starting_pivot = starting_pivot
        self.included_pivots = []
        self.starting_index = starting_index
        self.end_point = None
        self.first_condition = False
        self.second_condition = False

    def add_pivot(self, pivot_point, pivot_index):
        self.included_pivots.append((pivot_point, pivot_index))

    def calculate_end_point(self, max_index, filtered_data):
        if not self.included_pivots:
            return
        first_pvt = self.starting_pivot
        first_idx = self.starting_index
        last_pvt, last_idx = self.included_pivots[-1]

        dx = last_pvt.close - first_pvt.close
        dy = last_idx - first_idx
        end_price = last_pvt.close + 2 * dx
        end_index = last_idx + 2 * dy
        if end_index >= max_index:
            end_index = max_index - 1
        end_date = filtered_data.index[end_index]
        self.end_point = (end_date, end_price)