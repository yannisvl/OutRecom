import math

class Point2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point2d({self.x}, {self.y})"

    def __add__(self, other):
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2d(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point2d(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __truediv__(self, scalar):
        return Point2d(self.x / scalar, self.y / scalar)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)