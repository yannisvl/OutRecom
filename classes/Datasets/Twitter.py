import pandas as pd
from utils.Point2d import Point2d

from classes.Datasets.FLDataset import FLDataset


class Twitter(FLDataset):
    def __init__(self, file, keep_unique):
        super().__init__()
        self.points = []
        self.file = file
        self.unique = keep_unique
        self.read_data(file)

    def read_data(self, infile):
        rows = []
        with open(infile, 'r') as file:
            for line in file:
                parts = line.split('\t')
                coords = parts[1]
                coordinates = coords.split()
                rows.append([float(coordinates[0]), float(coordinates[1])])

        df = pd.DataFrame(rows)  

        unique_rows = df.drop_duplicates()
        if self.unique:
            df = unique_rows

        if self.random_sample:
            points_array = df.sample(n=self.max_entries, random_state=42).to_numpy()
        else:
            points_array = df.head(self.max_entries).to_numpy()
            
        self.points = [Point2d(x, y) for x, y in points_array]