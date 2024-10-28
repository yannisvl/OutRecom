from point2d import Point2D
from statistics import median
import math


def custom_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 0:
        return sorted_data[n // 2 - 1]
    else:
        return sorted_data[n // 2]
    

class CMP():
    def __init__(self, confidence):
        # Calling the constructor of the parent class
        super().__init__()
        self.confidence = confidence

    def solve(self, points, pred):
        # Step 1: Create c * n copies of pred
        copies = math.floor(self.confidence * len(points)) * [pred]

        # Add the copies to the initial set of points
        extended_points = points + copies

        # Step 3: Compute the median of x-values and y-values separately
        x_values = [point.x for point in extended_points]
        y_values = [point.y for point in extended_points]

        x_median = custom_median(x_values)
        y_median = custom_median(y_values)

        # Step 4: Combine the medians
        combined_median = Point2D(x_median, y_median)

        return combined_median