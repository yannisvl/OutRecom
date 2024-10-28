from classes.Algorithms.Weiszfeld import Weiszfeld
from classes.Algorithms.CMP import CMP
from classes.Datasets.Brightkite import Brightkite
from classes.Datasets.Gowalla import Gowalla
from classes.Datasets.Twitter import Twitter
from classes.Datasets.Autotel import Autotel
from classes.Datasets.Earthquake import Earthquake
from utils.Point2d import Point2d
import math
import numpy as np
import matplotlib.pyplot as plt


class FacilityLocation():
    def __init__(self, dataset, keep_unique, c):
        self.optAlg = Weiszfeld()
        self.predAlg = CMP(c)
        self.numPreds = 10 #per dimension
        self.datasetName = dataset

        if dataset == "Brightkite":
            self.dataset = Brightkite("datasets\loc-brightkite_totalCheckins.txt\Brightkite_totalCheckins.txt", keep_unique)
        elif dataset == "Gowalla":
            self.dataset = Gowalla("datasets\loc-gowalla_totalCheckins.txt\Gowalla_totalCheckins.txt", keep_unique)
        elif dataset == "Twitter":
            self.dataset = Twitter("datasets/timestamped_gps_coordinate.txt", keep_unique)
        elif dataset == "Autotel":
            self.dataset = Autotel("datasets/autotel.csv", keep_unique)
        elif dataset == "Earthquake":
            self.dataset = Earthquake("datasets/earthquake.csv", keep_unique)
        else:
            print("Invalid Dataset")
            exit()

    def create_predictions(self, instance):
        min_x = min(point.x for point in instance)
        max_x = max(point.x for point in instance)
        min_y = min(point.y for point in instance)
        max_y = max(point.y for point in instance)

        height = max_y - min_y
        width = max_x - min_x
        min_x -= 0.25*width
        max_x += 0.25*width
        min_y -= 0.25*height
        max_y += 0.25*height

        # Calculate the step sizes for the grid
        step_x = (max_x - min_x) / (self.numPreds - 1)
        step_y = (max_y - min_y) / (self.numPreds - 1)

        # Generate the grid points
        grid = []
        for i in range(self.numPreds):
            for j in range(self.numPreds):
                x = min_x + i * step_x
                y = min_y + j * step_y
                grid.append(Point2d(x, y))

        return grid
    
    def cost_sum(self, points, solution):
        distances = [x.distance_to(solution) for x in points]
        return sum(distances)
        
    def run_experiment(self):
        instance = self.dataset.points
        
        preds = self.create_predictions(instance)
        opt = self.optAlg.solve(instance, 100) #100 iterations
        opt_cost = self.cost_sum(instance, opt)
        print("Opt is ", opt, "with cost ", opt_cost)
        c = self.predAlg.confidence

        # Extract x and y coordinates for each type of point
        a_x = [point.x for point in instance]
        a_y = [point.y for point in instance]
        b_x = [point.x for point in preds]
        b_y = [point.y for point in preds]
        c_x = opt.x
        c_y = opt.y

        # Plotting
        plt.plot(a_x, a_y, 'ko')  # "a" points as black dots
        plt.plot(b_x, b_y, 'g+')  # "b" points as green crosses
        plt.plot(c_x, c_y, 'rx', markersize=10)  # "c" point as red X
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Plot with different point types')
        plt.grid(True)
        plt.savefig(self.datasetName+".png")

        data = np.zeros((len(preds), 5))
        for i, pred in enumerate(preds):
            sol = self.predAlg.solve(instance, pred)
            sol_cost = self.cost_sum(instance, sol)
            ratio = sol_cost / opt_cost
            
            ratio_error = self.cost_sum(instance, pred) / opt_cost
            ratio_bound = min(math.sqrt(2) * ratio_error, math.sqrt(2) + ratio_error)

            dist_error = self.dataset.max_entries * opt.distance_to(pred) / opt_cost
            dist_bound = math.sqrt(2*c**2 + 2) / (c + 1) + dist_error

            print("sol is", sol, "with cost", sol_cost)
            print("(ratio, ratio_bound, dist_bound) = ", ratio, ratio_bound, dist_bound)

            #save ratio, ratio_bound, dist_bound
            data[i,:] = (ratio, ratio_error, ratio_bound, dist_error, dist_bound)

            if math.sqrt(2*c**2 + 2) / (c + 1) + ratio_error < ratio:
                print("Conjecture does not hold!")

        np.savez("experiments/"+self.datasetName+"c"+str(self.predAlg.confidence)+"preds"+str(self.numPreds)+"unique"+str(self.dataset.unique)+'.npz', array1=data)