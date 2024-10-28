from utils.Point2d import Point2d
import numpy as np
from functools import reduce

class Weiszfeld():
    def __init__(self):
        super().__init__()

    def solve(self, points, iterations):
        num_points = len(points)

        # Initial guess for the median
        total_sum = reduce(lambda p1, p2: p1 + p2, points)
        median = total_sum / num_points

        for i in range(iterations):
            if i%10==0:
                print("iteration", i+1)

            # Calculate the weighted average
            weighted_sum = Point2d(0, 0)
            total_weight = 0.0

            for point in points:
                distance = point.distance_to(median)
                if distance != 0:  # Avoid division by zero
                    weight = 1.0 / distance
                    weighted_sum += weight * point
                    total_weight += weight

            # Update the median
            if total_weight != 0:  # Avoid division by zero
                new_median = weighted_sum / total_weight
                median = new_median

        print("Weiszfeld ended!")
        return median