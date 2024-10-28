import pandas as pd
from utils.Point2d import Point2d

from classes.Datasets.FLDataset import FLDataset


class Autotel(FLDataset):
    def __init__(self, file, keep_unique):
        super().__init__()
        self.points = []
        self.file = file
        self.unique = keep_unique
        self.read_data()

    def read_data(self):    
        
        df = pd.read_csv(self.file)
        df = df[['latitude', 'longitude']].dropna().astype(float)

        unique_rows = df.drop_duplicates()
        if self.unique:
            df = unique_rows
        
        if self.random_sample:
            points_array = df.sample(n=self.max_entries, random_state=42).to_numpy()
        else:
            points_array = df.head(self.max_entries).to_numpy()

        self.points = [Point2d(x, y) for x, y in points_array]