import pandas as pd
import numpy as np
import tarfile
import random


class Yahoo():
    def __init__(self, file):
        self.file = file
        self.min_values = []
        self.max_values = []
        self.non_zero_vals = []
        
        # bids = self.read_instance()
        # self.instance = self.transform_to_scheduling(bids)

        # np.random.seed(0)
        # self.instance = np.random.uniform(low=1, high=1000, size=(100, 100))
        
        # self.instance = np.array([[1,7,8,3],
        #                           [10,3,9,6],
        #                           [5,4,7,2]])

        # self.instance = np.array([[9,4,3,8,6],
        #                            [15,5,4,9,19],
        #                            [20,10,11,1,8]])

    def extract(self):
        with tarfile.open(self.file, 'r:gz') as tar:
            tar.extractall()

    def process_data(self):

        columns = ["day", "id", "rank", "keyphrase", "bid", "impressions", "clicks"]
        columns_to_keep = ["id", "keyphrase", "bid"]

        df = pd.read_csv(self.file, delimiter='\t', header=None, names=columns, usecols=columns_to_keep)

        # Group by 'day', 'id', and 'keyphrase', then find the row index with the maximum 'bid' value in each group
        max_bid_indices = df.groupby(['id', 'keyphrase'])['bid'].idxmax()

        # Filter the DataFrame using the indices of the rows with the maximum bid in each group
        df = df.loc[max_bid_indices]
        df.to_csv('filtered_yahoo.csv', index=False)

    def read_keyphrases(self):
        columns = ["day", "id", "rank", "keyphrase", "bid", "impressions", "clicks"]
        columns_to_keep = ["day", "id", "keyphrase", "bid"]
        self.df = pd.read_csv(self.file, delimiter='\t', header=None, names=columns, usecols=columns_to_keep)

        return list(self.df['keyphrase'].unique())
    
    def random_day(self, jobs):
        days = self.df[self.df['keyphrase'].isin(jobs)]
        return random.choice(days['day'].unique())
      
    def create_instance(self):
        n_machines = 100
        n_jobs = 100
        df = pd.read_csv('./datasets/Scheduling/filtered_yahoo.csv', names = ["id", "keyphrase", "bid"])

        bid_counts = df.groupby('id')['id'].count()
        max_bid_ids = bid_counts.sort_values(ascending=False).head(n_machines)
        filtered_df = df[df['id'].isin(max_bid_ids.index)]

        keyphrase_counts = filtered_df.groupby('keyphrase')['keyphrase'].count()
        top_keyphrases = keyphrase_counts.sort_values(ascending=False).head(n_jobs)
        filtered_df = filtered_df[filtered_df['keyphrase'].isin(top_keyphrases.index)]

        filtered_df.to_csv("instance.csv", index = True)

        return filtered_df


    def read_instance(self):
        df = pd.read_csv("./datasets/Scheduling/instance.csv", header = 0)
        pivot_df = df.pivot(index='id', columns='keyphrase', values='bid').fillna(0)
        array_2d = pivot_df.to_numpy()
        return array_2d


    def transform_to_scheduling(self, bids):
        self.non_zero_vals = np.where(bids==0, 0, 1)
        self.zero_vals = np.where(bids==0, 1, 0)

        # reverse bids per column
        rows, cols = bids.shape
        reversed_array = np.zeros((rows, cols), dtype=bids.dtype)
        for j in range(cols):
            non_zero_values = bids[:, j][bids[:, j] != 0]
            non_zero_indices = np.where(bids[:, j] != 0)[0]
            reversed_values = non_zero_values[::-1]
            reversed_array[non_zero_indices, j] = reversed_values

        maxVal = np.max(bids)
        processing_times = np.where(reversed_array > 0, reversed_array, maxVal**3)
        self.min_values = np.min(processing_times, axis=0)
        self.max_values = np.max(reversed_array, axis=0)

        return processing_times