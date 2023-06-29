import numpy as np


class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_average(self):
        if len(self.values) == 0:
            return None
        return np.mean(self.values)
